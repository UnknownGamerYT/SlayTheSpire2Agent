from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _end_turn_action, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster

from sts2sim import load_state, serialize, step
from sts2sim.engine import CardType
from sts2sim.mechanics.card_effects import card_effect_plan


def _source(card_id: str, description: str, **fields) -> dict:
    return dict(
        card_effect_plan(
            {
                "id": card_id,
                "type": "Attack" if fields.get("damage") else "Skill",
                "cost": 0,
                "target": "enemy"
                if fields.get("damage") or fields.get("powers_applied")
                else "self",
                "description": description,
                **fields,
            }
        ).card
    )


def _fight(card: dict):
    return _combat(_monster("VICTIM", {}), deck=(card, _card("filler", damage=0)))


def _no_stubs(state) -> None:
    assert not [event for event in state.combat.last_events if event.kind == "effect_stubbed"]


@pytest.mark.parametrize("poison", [0, 2])
def test_bubble_requires_existing_poison(poison: int) -> None:
    card = _source(
        "bubble_bubble",
        "If the enemy has [gold]Poison[/gold], apply 9 Poison.",
        powers_applied=({"power_key": "Poison", "amount": 9},),
    )
    state = _fight(card)
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"poison": poison}})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster,),
                }
            )
        }
    )
    state = _play_first_card(state, "bubble_bubble")
    assert state.combat.monsters[0].statuses.get("poison", 0) == poison + (9 if poison else 0)
    _no_stubs(state)


@pytest.mark.parametrize("intent,weak", [("Attack", 1), ("Defend", 0), ("Attack + Buff", 1)])
def test_go_for_the_eyes_weak_requires_attack_intent(intent: str, weak: int) -> None:
    card = _source(
        "go_for_the_eyes",
        "Deal 3 damage. If the enemy intends to attack, apply 1 Weak.",
        damage=3,
        powers_applied=({"power_key": "Weak", "amount": 1},),
    )
    state = _fight(card)
    monster = state.combat.monsters[0].model_copy(update={"intent": intent})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster,),
                }
            )
        }
    )
    state = _play_first_card(state, "go_for_the_eyes")
    assert state.combat.monsters[0].statuses.get("weak", 0) == weak
    _no_stubs(state)


@pytest.mark.parametrize("vulnerable,hp", [(0, 92), (1, 76)])
def test_dismantle_extra_hit_requires_vulnerable(vulnerable: int, hp: int) -> None:
    state = _fight(
        _source("dismantle", "Deal 8 damage. If the enemy is Vulnerable, hits twice.", damage=8)
    )
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"vulnerable": vulnerable}})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster,),
                }
            )
        }
    )
    state = _play_first_card(state, "dismantle")
    assert state.combat.monsters[0].hp == hp
    _no_stubs(state)


@pytest.mark.parametrize("played,expected_draw", [(0, 1), (2, 1), (3, 0)])
def test_ftl_uses_cards_played_before_this_play(played: int, expected_draw: int) -> None:
    state = _fight(
        _source(
            "ftl",
            "Deal 5 damage. If you have played fewer than 3 cards this turn, draw 1 card.",
            damage=5,
            cards_draw=1,
        )
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": tuple(card for card in state.combat.hand if card.card_id == "ftl"),
                    "draw_pile": (filler,),
                    "cards_played_this_turn": tuple(f"prior:{i}" for i in range(played)),
                }
            )
        }
    )
    state = _play_first_card(state, "ftl")
    assert len(state.combat.hand) == expected_draw
    _no_stubs(state)


@pytest.mark.parametrize("has_attack", [False, True])
def test_impatience_checks_remaining_hand(has_attack: bool) -> None:
    state = _fight(
        _source("impatience", "If you have no Attacks in your Hand, draw 2 cards.", cards_draw=2)
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    hand = tuple(card for card in state.combat.hand if card.card_id == "impatience")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": hand + ((filler,) if has_attack else ()),
                    "draw_pile": (filler.model_copy(update={"instance_id": "drawn"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "impatience")
    assert [card.instance_id for card in state.combat.hand] == (
        [filler.instance_id] if has_attack else ["drawn"]
    )
    _no_stubs(state)


@pytest.mark.parametrize("empty", [False, True])
def test_restlessness_checks_hand_once_before_drawing_and_gaining_energy(empty: bool) -> None:
    state = _fight(
        _source(
            "restlessness",
            "If your Hand is empty, draw 2 cards and gain [energy:2].",
            cards_draw=2,
            energy_gain=2,
        )
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": tuple(
                        card for card in state.combat.hand if card.card_id == "restlessness"
                    )
                    + (() if empty else (filler,)),
                    "draw_pile": (filler.model_copy(update={"instance_id": "drawn"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "restlessness")
    assert state.combat.player.energy == 3 + (2 if empty else 0)
    assert len(state.combat.draw_pile) == (0 if empty else 1)
    _no_stubs(state)


@pytest.mark.parametrize("count,hp", [(2, 100), (3, 83)])
def test_pacts_end_threshold_does_not_become_accidental_draw(count: int, hp: int) -> None:
    state = _fight(
        _source(
            "pacts_end",
            "If you have 3 or more cards in your Exhaust Pile, deal 17 damage to ALL enemies.",
            damage=17,
            cards_draw=3,
            target="all_enemies",
        )
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "exhaust_pile": tuple(
                        filler.model_copy(update={"instance_id": f"old:{i}"}) for i in range(count)
                    ),
                    "draw_pile": (filler.model_copy(update={"instance_id": "drawn"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "pacts_end")
    assert state.combat.monsters[0].hp == hp
    assert len(state.combat.draw_pile) == 1
    _no_stubs(state)


@pytest.mark.parametrize("card_type,expected_block", [(CardType.SKILL, 3), (CardType.ATTACK, 0)])
def test_escape_plan_draws_before_testing_the_actual_card_type(
    card_type, expected_block: int
) -> None:
    state = _fight(
        _source("escape_plan", "Draw 1 card. If you draw a Skill, gain 3 Block.", block=3)
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    drawn = filler.model_copy(update={"instance_id": "drawn", "type": card_type})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (drawn,),
                }
            )
        }
    )
    state = _play_first_card(state, "escape_plan")
    assert any(card.instance_id == "drawn" for card in state.combat.hand)
    assert state.combat.player.block == expected_block
    _no_stubs(state)


def test_fetch_draws_only_on_first_play_of_the_physical_card_each_turn() -> None:
    state = _fight(
        _source(
            "fetch",
            "Osty deals 3 damage. If this is the first time this card "
            "has been played this turn, draw 1 card.",
            damage=3,
            cards_draw=1,
        )
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {**state.combat.metadata, "osty": {
                        "hp": 1, "max_hp": 1, "alive": True,
                    }},
                    "draw_pile": (filler.model_copy(update={"instance_id": "drawn"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "fetch")
    assert not state.combat.draw_pile
    fetch = next(card for card in state.combat.discard_pile if card.card_id == "fetch")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": state.combat.hand + (fetch,),
                    "discard_pile": (),
                    "draw_pile": (filler.model_copy(update={"instance_id": "drawn2"}),),
                }
            )
        }
    )
    state = _play_first_card(load_state(serialize(state)), "fetch")
    assert state.combat.draw_pile[0].instance_id == "drawn2"
    _no_stubs(state)


def test_exhaust_condition_is_live_persistent_and_expires_with_turn() -> None:
    ritual = _source(
        "forgotten_ritual", "If you Exhausted a card this turn, gain [energy:3].", energy_gain=3
    )
    sacrifice = {**_card("sacrifice", damage=0), "exhausts": True}
    state = _combat(_monster("VICTIM", {}), deck=(ritual, sacrifice))
    state = _play_first_card(state, "sacrifice")
    assert state.combat.metadata["last_exhaust_turn"] == state.combat.turn
    state = _play_first_card(load_state(serialize(state)), "forgotten_ritual")
    assert state.combat.player.energy == 6
    state = step(state, _end_turn_action(state))
    state = _play_first_card(state, "forgotten_ritual")
    assert state.combat.player.energy == 3
    _no_stubs(state)


@pytest.mark.parametrize("exhausted,block", [(False, 8), (True, 16)])
def test_evil_eye_bonus_checks_exhaust_event_this_turn(exhausted: bool, block: int) -> None:
    card = _source(
        "evil_eye",
        "Gain 8 Block. Gain another 8 Block if you have Exhausted a card this turn.",
        block=8,
    )
    sacrifice = {**_card("sacrifice", damage=0), "exhausts": True}
    state = _combat(_monster("VICTIM", {}), deck=(card, sacrifice))
    if exhausted:
        state = _play_first_card(state, "sacrifice")
    state = _play_first_card(state, "evil_eye")
    assert state.combat.player.block == block
    _no_stubs(state)


@pytest.mark.parametrize(
    "doom,artifact,block", [(False, False, 6), (True, False, 18), (True, True, 6)]
)
def test_deaths_door_requires_successfully_applied_doom(
    doom: bool,
    artifact: bool,
    block: int,
) -> None:
    card = _source(
        "deaths_door",
        "Gain 6 Block. If you applied Doom this turn, gain Block 2 additional times.",
        block=6,
    )
    doom_card = _source(
        "doom", "Apply 2 Doom.", powers_applied=({"power_key": "Doom", "amount": 2},)
    )
    state = _combat(_monster("VICTIM", {"ARTIFACT": 1} if artifact else {}), deck=(card, doom_card))
    if doom:
        state = _play_first_card(state, "doom")
    state = _play_first_card(state, "deaths_door")
    assert state.combat.player.block == block
    _no_stubs(state)


def test_forgotten_ritual_gives_no_energy_before_any_exhaust() -> None:
    state = _fight(
        _source(
            "forgotten_ritual", "If you Exhausted a card this turn, gain [energy:3].", energy_gain=3
        )
    )
    state = _play_first_card(state, "forgotten_ritual")
    assert state.combat.player.energy == 3
    _no_stubs(state)


def test_escape_plan_no_draw_does_not_grant_block() -> None:
    state = _fight(
        _source("escape_plan", "Draw 1 card. If you draw a Skill, gain 3 Block.", block=3)
    )
    player = state.combat.player.model_copy(update={"statuses": {"no_draw_this_turn": 1}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = _play_first_card(state, "escape_plan")
    assert state.combat.player.block == 0
    _no_stubs(state)


def test_fetch_replay_does_not_repeat_first_physical_play_draw() -> None:
    fetch = _source(
        "fetch",
        "Osty deals 3 damage. If this is the first time this card "
        "has been played this turn, draw 1 card.",
        damage=3,
        cards_draw=1,
    )
    fetch["enchantments"] = ({"keyword": "replay", "amount": 1},)
    state = _fight(fetch)
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {**state.combat.metadata, "osty": {
                        "hp": 1, "max_hp": 1, "alive": True,
                    }},
                    "draw_pile": tuple(
                        filler.model_copy(update={"instance_id": f"drawn:{i}"}) for i in range(2)
                    ),
                }
            )
        }
    )
    state = _play_first_card(state, "fetch")
    assert len(state.combat.draw_pile) == 1
    _no_stubs(state)


def test_fetch_with_missing_osty_neither_attacks_nor_draws() -> None:
    state = _fight(_source("fetch", "Osty deals 3 damage. Draw 1 card.",
                           damage=3, cards_draw=1))
    before_hp = state.combat.monsters[0].hp
    draw_ids = tuple(c.instance_id for c in state.combat.draw_pile)
    state = _play_first_card(state, "fetch")
    assert state.combat.monsters[0].hp == before_hp
    assert tuple(c.instance_id for c in state.combat.draw_pile) == draw_ids
    assert not any(e.kind == "cards_drawn" for e in state.combat.last_events)
    _no_stubs(state)
