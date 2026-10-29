from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_combat_engine_monsters import _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster

from sts2sim import legal_actions, load_state, serialize
from sts2sim.engine import OrbState

SOURCE = {
    row["id"]: row
    for row in json.loads(
        (Path(__file__).parents[1] / "fixtures" / "x_cost_source.json").read_text()
    )
}


def _fight(
    card_id: str,
    *,
    energy: int,
    stars: int = 0,
    chemical: bool = False,
    upgraded: bool = False,
    replay: bool = False,
):
    card = {**SOURCE[card_id], "upgraded": upgraded}
    if replay:
        card["enchantments"] = ({"keyword": "replay", "amount": 1},)
    monsters = tuple(_monster(name, {}, damage=0) for name in ("FIRST", "SECOND"))
    for monster in monsters:
        monster.update(min_hp=1000, max_hp=1000)
    state = _combat(*monsters, deck=(card, _card("filler", damage=1)))
    player = state.combat.player.model_copy(update={"energy": energy, "resources": {"star": stars}})
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "relics": ("chemical_x",) if chemical else (),
            "player": player,
            "combat": state.combat.model_copy(
                update={
                    "player": player,
                    "orb_slots": 10,
                    "orbs": (OrbState(orb_id="lightning"),),
                    "hand": tuple(card for card in state.combat.hand if card.card_id != "filler"),
                    "draw_pile": tuple(
                        filler.model_copy(update={"instance_id": f"filler:{i}"}) for i in range(12)
                    )
                    if card_id == "CASCADE"
                    else (),
                }
            ),
        }
    )
    return state


def _damage(state) -> int:
    return sum(1000 - monster.hp for monster in state.combat.monsters)


@pytest.mark.parametrize("card_id", tuple(SOURCE))
@pytest.mark.parametrize(
    "x,chemical,upgraded",
    [
        (0, False, False),
        (0, False, True),
        (0, True, False),
        (2, False, False),
        (2, True, False),
        (2, True, True),
    ],
)
def test_every_stable_x_card_uses_its_resource_and_chemical_bonus(
    card_id: str,
    x: int,
    chemical: bool,
    upgraded: bool,
) -> None:
    star_card = card_id == "STARDUST"
    state = _fight(
        card_id,
        energy=7 if star_card else x,
        stars=x if star_card else 7,
        chemical=chemical,
        upgraded=upgraded,
    )
    card = state.combat.hand[0]
    assert card.cost == (0 if star_card else -1)
    assert any(action.card_instance_id == card.instance_id for action in legal_actions(state))
    result = _play_first_card(state, card_id.lower())
    assert result.combat.player.energy == (7 if star_card else 0)
    assert result.combat.player.resources["star"] == (0 if star_card else 7)
    assert not [event for event in result.combat.last_events if event.kind.endswith("stubbed")]
    value = x + (2 if chemical else 0)
    extra = int(upgraded)
    if card_id in {"ERADICATE", "SKEWER", "VOLLEY", "STARDUST"}:
        damage = {"ERADICATE": (11, 14), "SKEWER": (8, 11), "VOLLEY": (10, 14), "STARDUST": (5, 7)}[
            card_id
        ][upgraded]
        assert _damage(result) == value * damage
    elif card_id == "WHIRLWIND":
        assert _damage(result) == value * (8 if upgraded else 5) * 2
    elif card_id == "HEAVENLY_DRILL":
        hits = value * 2 if value >= 4 else value
        assert _damage(result) == hits * (10 if upgraded else 8)
    elif card_id == "MALAISE":
        assert result.combat.monsters[0].statuses.get("strength", 0) == -(value + extra)
        assert result.combat.monsters[0].statuses.get("weak", 0) == value + extra
    elif card_id == "DIRGE":
        assert result.combat.player.resources.get("summon", 0) == value * (4 if upgraded else 3)
        assert sum(card.card_id == "soul" for card in result.combat.draw_pile) == value
    elif card_id == "TEMPEST":
        assert len(result.combat.orbs) == 1 + value + extra
    elif card_id == "MULTI_CAST":
        assert _damage(result) == (value + extra) * 8
        assert len(result.combat.orbs) == (1 if value + extra == 0 else 0)
    elif card_id == "CASCADE":
        assert _damage(result) == value + extra
        assert len(result.combat.draw_pile) == 12 - value - extra


@pytest.mark.parametrize(
    "card_id,energy,stars,damage",
    [
        ("SKEWER", 0, 0, 32),
        ("SKEWER", 2, 0, 64),
        ("HEAVENLY_DRILL", 2, 0, 128),
        ("STARDUST", 7, 2, 40),
    ],
)
def test_replays_preserve_original_x_and_do_not_apply_chemical_twice(
    card_id: str,
    energy: int,
    stars: int,
    damage: int,
) -> None:
    state = _fight(card_id, energy=energy, stars=stars, chemical=True, replay=True)
    restored = load_state(serialize(state))
    results = [_play_first_card(candidate, card_id.lower()) for candidate in (state, restored)]
    assert serialize(results[0]) == serialize(results[1])
    assert _damage(results[0]) == damage
    assert results[0].combat.player.energy == (7 if card_id == "STARDUST" else 0)
    assert results[0].combat.player.resources["star"] == 0


@pytest.mark.parametrize(
    "card_id,energy,stars,damage",
    [
        ("SKEWER", 3, 0, 40),
        ("STARDUST", 7, 2, 20),
    ],
)
def test_physical_autoplay_uses_current_x_resource_without_spending_it(
    card_id: str,
    energy: int,
    stars: int,
    damage: int,
) -> None:
    state = _fight(card_id, energy=energy, stars=stars, chemical=True)
    played = state.combat.hand[0]
    trigger = state.master_deck[1].model_copy(
        update={
            "card_id": "autoplay",
            "instance_id": "trigger",
            "effects": {"play_top_card": {"amount": 1}},
        }
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (trigger,),
                    "draw_pile": (played,),
                }
            )
        }
    )
    state = _play_first_card(state, "autoplay")
    assert _damage(state) == damage
    assert state.combat.player.energy == energy
    assert state.combat.player.resources["star"] == stars
    assert not [
        event for event in state.combat.last_events if event.kind == "player_resource_spent"
    ]


def test_cascade_child_x_uses_energy_gained_by_earlier_autoplayed_card() -> None:
    state = _fight("CASCADE", energy=2, chemical=True)
    template = state.master_deck[1]
    generator = template.model_copy(
        update={
            "card_id": "generator",
            "instance_id": "generator",
            "effects": {"energy": 3},
        }
    )
    skewer = _fight("SKEWER", energy=0).combat.hand[0].model_copy(update={"instance_id": "child"})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (generator, skewer),
                }
            )
        }
    )
    state = _play_first_card(state, "cascade")
    assert state.combat.player.energy == 3
    assert _damage(state) == 40


def test_chemical_x_does_not_turn_effect_bonus_into_energy_spent_for_danse() -> None:
    state = _fight("SKEWER", energy=0, chemical=True)
    player = state.combat.player.model_copy(update={"statuses": {"dansemacabre": 4}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = _play_first_card(state, "skewer")
    assert _damage(state) == 16
    assert state.combat.player.energy == 0
    assert state.combat.player.block == 0


def test_void_form_does_not_waive_stardusts_x_star_payment() -> None:
    state = _fight("STARDUST", energy=7, stars=2, chemical=True)
    player = state.combat.player.model_copy(update={"statuses": {"voidform": 1}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = _play_first_card(state, "stardust")
    assert state.combat.player.resources["star"] == 0
    assert state.combat.player.energy == 7
    assert _damage(state) == 20
