from __future__ import annotations

import pytest
from test_named_monster_scripts import _combat
from test_remaining_monster_scripts import _end

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine.transitions import _card_from_spec


def _card(card_id, *, effects=None, **fields):
    source = {"id": card_id, "instance_id": f"test:{card_id}",
              "type": "Skill", "target": "Self", "cost": 0, **fields}
    if effects is not None:
        source["effects"] = effects
    return _card_from_spec(source, 1)


def _fight(*cards, allies=()):
    state = _combat("TEST", moves=({"id": "WAIT", "intent": "Sleep"},))
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": cards, "draw_pile": tuple(_card(f"filler_{i}", block=1) for i in range(20)),
        "metadata": {**state.combat.metadata, "allies": allies},
    })})


def _play(state, card_id):
    instance_id = next(c.instance_id for c in state.combat.hand if c.card_id == card_id)
    return step(state, next(a for a in legal_actions(state)
                           if a.type == "play_card" and a.card_instance_id == instance_id))


def test_demesne_grants_energy_and_additional_draw_only_at_each_turn_start() -> None:
    card = _card("DEMESNE", type="Power", description=(
        "At the start of your turn, gain [energy:1] and draw 1 additional card."
    ))
    state = _play(_fight(card), "demesne")
    assert state.combat.player.energy == 3
    assert state.combat.hand == ()
    for _ in range(2):
        state = _end(load_state(serialize(state)))
        assert state.combat.player.energy == 4
        assert len(state.combat.hand) == 6


def test_subroutine_triggers_once_for_later_power_plays_and_not_skills() -> None:
    cards = (
        _card("SUBROUTINE", type="Power", description=(
            "Whenever you play a Power, gain [energy:1]."
        )),
        _card("power", type="Power", effects={"noop": True}),
        _card("skill", effects={"block": 1}),
    )
    state = _play(_fight(*cards), "subroutine")
    assert state.combat.player.energy == 3
    state = _play(state, "power")
    assert state.combat.player.energy == 4
    state = _play(state, "skill")
    assert state.combat.player.energy == 4


@pytest.mark.parametrize("played_first", [False, True])
def test_drum_of_battle_gains_energy_when_its_physical_card_is_exhausted(played_first) -> None:
    drum = _card("DRUM_OF_BATTLE", description=(
        "Draw 2 cards. When this card is [gold]Exhausted[/gold], gain [energy:2]."
    ))
    exhauster = _card("exhauster", effects={"exhaust_hand": {"mode": "all"}})
    state = _fight(drum, exhauster)
    if played_first:
        state = _play(state, "drum_of_battle")
        state = state.model_copy(update={"combat": state.combat.model_copy(update={
            "hand": (drum, exhauster), "discard_pile": (),
        })})
    state = _play(state, "exhauster")
    assert state.combat.player.energy == 5
    assert [c.instance_id for c in state.combat.exhaust_pile] == [drum.instance_id]


def test_echo_form_replays_only_the_first_card_each_turn_and_moves_one_physical_card() -> None:
    echo = _card("ECHO_FORM", type="Power", description=(
        "The first card you play each turn is played an extra time."
    ))
    state = _end(_play(_fight(echo), "echo_form"))
    attack = _card("attack", type="Attack", target="AnyEnemy", effects={"damage": 4})
    second = attack.model_copy(update={"instance_id": "second", "card_id": "second"})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (attack, second),
    })})
    state = _play(state, "attack")
    assert state.combat.monsters[0].hp == 92
    assert sum(c.instance_id == attack.instance_id for c in state.combat.discard_pile) == 1
    state = _play(state, "second")
    assert state.combat.monsters[0].hp == 88
    state = _end(load_state(serialize(state)))
    skill = _card("skill", effects={"block": 2})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (skill,)})})
    state = _play(state, "skill")
    assert state.combat.player.block == 4


@pytest.mark.parametrize("card_id,damage,keyword,trigger", [
    ("HOWL_FROM_BEYOND", 16, (), "end"),
    ("BOMBARDMENT", 18, ("Exhaust",), "start"),
])
def test_exhaust_pile_passives_play_without_prior_play_and_use_normal_destination(
    card_id, damage, keyword, trigger,
) -> None:
    description = (
        f"Deal {damage} damage{' to ALL enemies' if trigger == 'end' else ''}. "
        f"At the {'end' if trigger == 'end' else 'start'} of your turn, "
        "if this is in your [gold]Exhaust Pile[/gold], play it."
    )
    card = _card(card_id, type="Attack", target="AllEnemies" if trigger == "end" else "AnyEnemy",
                 damage=damage, description=description, keywords=keyword, keywords_key=keyword)
    state = _fight()
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "exhaust_pile": (card,), "draw_per_turn": 0,
    })})
    state = _end(load_state(serialize(state)))
    assert state.combat.monsters[0].hp == 100 - damage
    assert state.combat.player.energy == 3
    destination = state.combat.exhaust_pile if keyword else state.combat.discard_pile
    assert sum(c.instance_id == card.instance_id for c in destination) == 1
    state = _end(state)
    assert state.combat.monsters[0].hp == 100 - damage * (2 if keyword else 1)


@pytest.mark.parametrize("on_top,expected_hp", [(True, 78), (False, 68)])
def test_i_am_invincible_only_plays_from_top_draw_before_enemy_attack(on_top, expected_hp) -> None:
    card = _card("I_AM_INVINCIBLE", block=10, description=(
        "Gain 10 [gold]Block[/gold]. At the end of your turn, "
        "if this is on top of your [gold]Draw Pile[/gold], play it."
    ))
    state = _combat("TEST", moves=({"id": "HIT", "damage": {"normal": 12}},))
    filler = _card("filler", block=1)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (), "draw_pile": (card, filler) if on_top else (filler, card), "draw_per_turn": 0,
    })})
    state = _end(state)
    assert state.combat.player.hp == expected_hp
    assert (card.instance_id in {c.instance_id for c in state.combat.discard_pile}) is on_top


def test_beacon_of_hope_updates_each_ally_with_half_actual_block() -> None:
    beacon = _card("BEACON_OF_HOPE", type="Power", description=(
        "Whenever you gain [gold]Block[/gold] on your turn, "
        "other players gain half that much [gold]Block[/gold]."
    ))
    state = _fight(beacon, _card("defend", block=7), allies=(
        {"id": "a", "block": 1}, {"id": "b", "block": 4},
    ))
    state = _play(_play(state, "beacon_of_hope"), "defend")
    assert state.combat.player.block == 7
    assert [a["block"] for a in state.combat.metadata["allies"]] == [4, 7]


def test_hammer_time_updates_each_ally_forge_without_changing_player_amount() -> None:
    hammer = _card("HAMMER_TIME", type="Power", description=(
        "Whenever you [gold]Forge[/gold], all allies [gold]Forge[/gold] as well."
    ))
    forge = _card("forge", effects={"player_resource": {"resource": "forge", "amount": 5}})
    state = _fight(hammer, forge, allies=(
        {"id": "a", "resources": {"forge": 2, "star": 3}}, {"id": "b"},
    ))
    state = _play(_play(load_state(serialize(state)), "hammer_time"), "forge")
    assert state.combat.player.resources["forge"] == 5
    assert [a["resources"]["forge"] for a in state.combat.metadata["allies"]] == [7, 5]
    assert state.combat.metadata["allies"][0]["resources"]["star"] == 3


def test_lethal_exhaust_autoplay_ends_combat_before_later_end_turn_damage() -> None:
    howl = _card("HOWL_FROM_BEYOND", type="Attack", target="AllEnemies", damage=16)
    state = _fight()
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "exhaust_pile": (howl,),
        "monsters": (state.combat.monsters[0].model_copy(update={"hp": 16}),),
        "player": state.combat.player.model_copy(update={"hp": 1, "statuses": {"constrict": 3}}),
    })})
    state = _end(state)
    assert state.phase.value == "reward"
    assert state.player.hp == 1


def test_nested_generated_cards_use_card_creation_relic_hooks() -> None:
    power = _card("test_generator", type="Power", effects={"combat_trigger": {
        "trigger": "turn_start", "duration": "combat", "effects": ({"add_card_to_hand": {
            "card": {"id": "SHIV", "type": "Attack", "cost": 0, "damage": 4,
                     "target": "AnyEnemy"},
        }},),
    }})
    state = _fight(power)
    state = state.model_copy(update={"relics": ("regalite",)})
    state = _end(_play(state, "test_generator"))
    assert any(card.card_id == "shiv" for card in state.combat.hand)
    assert state.combat.player.block == 2
