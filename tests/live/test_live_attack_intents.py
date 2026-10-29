"""Source attack labels are final damage, not a raw amount to modify twice."""

from __future__ import annotations

from copy import deepcopy

import pytest
from live.test_live_ppo_policy import _combat

from sts2sim import load_state, serialize, step
from sts2sim.engine import Action
from sts2sim.live_intent_fields import import_monster_intents
from sts2sim.live_policy import import_policy_combat


def _import(raw):
    # Exercise the public importer, including its production helper wiring.
    return import_policy_combat(raw)


@pytest.mark.parametrize(
    "base,enemy,player,label,hits,total",
    [
        (12, [("STRENGTH_POWER", 3)], [], "15", 1, 15),
        (13, [("WEAK_POWER", 1)], [], "9", 1, 9),
        (13, [], [("VULNERABLE_POWER", 1)], "19", 1, 19),
        (13, [("WEAK_POWER", 1)], [("VULNERABLE_POWER", 1)], "14", 1, 14),
        (5, [("STRENGTH_POWER", 3)], [], "8x2", 2, 16),
        (2.5, [("STRENGTH_POWER", 1)], [], "3×2", 2, 6),
        (24, [], [("INTANGIBLE_POWER", 1)], "1", 1, 1),
        (0, [("STRENGTH_POWER", 3)], [], "3", 1, 3),
    ],
)
def test_raw_current_attack_modifiers_apply_once_in_actual_end_turn_and_reload(
    base,
    enemy,
    player,
    label,
    hits,
    total,
):
    raw = _combat()
    raw["battle"]["enemies"][0].update(
        status=[{"id": key, "amount": amount} for key, amount in enemy],
        intents=[{"type": "Attack", "label": label, "base_damage": base, "repeats": hits}],
    )
    raw["player"]["status"] = [{"id": key, "amount": amount} for key, amount in player]
    state = _import(raw)
    assert state.combat.monsters[0].intent_damage == total
    before = serialize(state)
    action = Action(type="end_turn")
    result = step(state, action)
    assert state.player.hp - result.player.hp == total
    assert serialize(result) == serialize(step(load_state(before), action))
    assert serialize(state) == before


def test_playing_weak_after_capture_refreshes_final_intent_from_original_base():
    raw = _combat()
    card = {
        "id": "SCARE",
        "index": 0,
        "is_upgraded": False,
        "target_type": "AllEnemies",
        "cost": "0",
        "can_play": True,
    }
    raw["player"]["hand"] = [card]
    raw["player"]["deck"] = [deepcopy(card)]
    raw["battle"]["enemies"][0]["intents"] = [
        {"type": "Attack", "label": "13", "base_damage": 13, "repeats": 1},
    ]
    state = _import(raw)
    state = step(state, Action(type="play_card", card_instance_id=state.combat.hand[0].instance_id))
    assert state.combat.monsters[0].statuses["weak"] == 1
    assert state.combat.monsters[0].intent_damage == 9
    result = step(state, Action(type="end_turn"))
    assert state.player.hp - result.player.hp == 9


def test_rounded_modifier_requires_base_even_when_display_does_not_reveal_it():
    raw = _combat()
    state = import_policy_combat(raw)
    monster = raw["battle"]["enemies"][0]
    monster.update(
        status=[{"id": "WEAK_POWER", "amount": 1}], intents=[{"type": "Attack", "label": "0"}]
    )
    with pytest.raises(ValueError, match="explicit base_damage"):
        import_monster_intents(monster, state.combat.monsters[0], state.combat, state.relics)


def test_explicit_base_repeats_and_displayed_hooks_must_agree():
    raw = _combat()
    state = import_policy_combat(raw)
    monster = raw["battle"]["enemies"][0]
    for intent, error in [
        ({"type": "Attack", "label": "12", "base_damage": 13}, "contradict"),
        ({"type": "Attack", "label": "12x2", "base_damage": 12, "repeats": 1}, "repeats"),
        ({"type": "Attack", "label": "12", "base_damage": True}, "finite numeric"),
    ]:
        monster["intents"] = [intent]
        with pytest.raises(ValueError, match=error):
            import_monster_intents(monster, state.combat.monsters[0], state.combat, state.relics)


def test_current_intent_metadata_keeps_only_declared_visible_fields():
    raw = _combat()
    raw["battle"]["enemies"][0]["intents"][0].update(
        base_damage=12,
        repeats=1,
        private_rng={"seed": "hidden"},
        next_move="hidden",
    )
    state = _import(raw)
    assert state.combat.monsters[0].metadata["live_visible_intents"] == [
        {"type": "Attack", "label": "12", "base_damage": 12, "repeats": 1},
    ]
