from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_card_play_status_runtime import _reload, _state
from combat.test_glam import _play

from sts2sim import legal_actions, serialize, step
from sts2sim.engine import CardEnchantment
from sts2sim.engine.transitions import _card_from_spec
from sts2sim.mechanics.source_debuff_rules import SOURCE_DEBUFF_POWER_IDS

SOURCE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "source_debuff_damage.json").read_text()
)
CARDS = {row["id"]: row for row in SOURCE["cards"]}


def _fight(card_id, statuses, *, upgraded=False, replay=False):
    card = _card_from_spec({
        **CARDS[card_id], "upgraded": upgraded, "instance_id": card_id.lower(),
    }, 0)
    if replay:
        card = card.model_copy(update={
            "enchantments": (CardEnchantment(keyword="replay", amount=1),),
        })
    state = _state(card)
    monster = state.combat.monsters[0].model_copy(update={"statuses": statuses})
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})


def test_rend_taxonomy_matches_all_cached_stable_debuff_power_types():
    source_debuffs = {
        row["id"] for row in SOURCE["powers"] if row["type"] == "Debuff"
    }
    assert source_debuffs == SOURCE_DEBUFF_POWER_IDS


@pytest.mark.parametrize("upgraded,per_stack", [(False, 2), (True, 3)])
@pytest.mark.parametrize("stacks", [0, 1, 2])
def test_bully_adds_damage_for_every_vulnerable_stack_before_vulnerability_multiplier(
    upgraded, per_stack, stacks,
):
    state = _fight("BULLY", {"vulnerable": stacks}, upgraded=upgraded)
    state = _play(_reload(state), "bully")
    damage = int((4 + stacks * per_stack) * (1.5 if stacks else 1))
    assert state.combat.monsters[0].hp == 100 - damage


@pytest.mark.parametrize("upgraded,base,per_type", [(False, 15, 5), (True, 18, 8)])
@pytest.mark.parametrize("statuses,count", [
    ({}, 0), ({"weak": 1}, 1), ({"weak": 2}, 1),
    ({"weak": 2, "poison": 2}, 2),
    ({"weak": 0, "poison": 0}, 0),
    ({"strength": 3, "enrage": 2, "unknown_buff": 3}, 0),
    ({"weak": 2, "strength": 3, "enrage": 2}, 1),
    ({"strength": -2}, 1),
    ({"temporary_strength": -4, "strength": 5}, 0),
    ({"temporary_strength": -4, "strength": 2}, 1),
    ({"strength": -2, "darkshackles": 2, "piercing_wail": 3}, 1),
    ({"imbalanced": 1, "debilitate": 2}, 2),
    ({"WeakPower": 1, "weak": 1}, 1),
    ({"shriek": -1}, 1),
])
def test_rend_counts_unique_source_debuffs_excluding_buffs_and_strength_recovery_markers(
    upgraded, base, per_type, statuses, count,
):
    # Stable Rend+: https://sts2.untapped.gg/en/cards/colorless/rend
    # The wiki currently shows beta's 10/12 base, with an explicit beta warning.
    # Its notes document effective negative Strength as one debuff, excluding
    # the temporary recovery powers themselves:
    # https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Rend#Notes
    state = _fight("REND", statuses, upgraded=upgraded)
    state = _play(_reload(state), "rend")
    assert state.combat.monsters[0].hp == 100 - base - per_type * count


@pytest.mark.parametrize("card_id,statuses,damage", [
    ("BULLY", {"vulnerable": 2}, 12),
    ("REND", {"weak": 2, "poison": 2}, 25),
])
def test_target_scaling_uses_selected_enemy_and_replays_after_save_reload(
    card_id, statuses, damage,
):
    state = _fight(card_id, {}, replay=True)
    first = state.combat.monsters[0]
    target = first.model_copy(update={"monster_id": "selected", "statuses": statuses})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, target),
    })})

    def play(candidate):
        return step(candidate, next(
            action for action in legal_actions(candidate)
            if action.type == "play_card" and action.target_id == "selected"
        ))

    result, restored = (play(candidate) for candidate in (state, _reload(state)))
    assert serialize(result) == serialize(restored)
    assert [monster.hp for monster in result.combat.monsters] == [100, 100 - 2 * damage]
