from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _end_turn_action, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardInstance
from sts2sim.mechanics.card_effects import card_effect_plan


def _source(card_id: str, text: str, **fields) -> dict:
    return dict(
        card_effect_plan(
            {
                "id": card_id,
                "type": "Attack",
                "cost": 0,
                "target": "enemy",
                "description": text,
                **fields,
            }
        ).card
    )


def _fight(spec: dict):
    return _combat(_monster("target", {}, damage=3), deck=(spec,))


@pytest.mark.parametrize(
    "card_id,text,zone",
    [
        (
            "CLASH",
            "Can only be played if every card in your Hand is an Attack. Deal 14 damage.",
            "hand",
        ),
        (
            "GRAND_FINALE",
            "Can only be played if there are no cards in your Draw Pile. "
            "Deal 60 damage to ALL enemies.",
            "draw_pile",
        ),
    ],
)
def test_restricted_attack_is_legal_only_when_condition_holds(card_id, text, zone):
    state = _fight(_source(card_id, text, damage=14))
    assert any(action.type == "play_card" for action in legal_actions(state))
    extra = CardInstance(instance_id="extra", card_id="defend", type="skill", cost=0)
    combat = state.combat.model_copy(update={zone: getattr(state.combat, zone) + (extra,)})
    state = state.model_copy(update={"combat": combat})
    assert not any(action.card_instance_id == "card_001" for action in legal_actions(state))


@pytest.mark.parametrize("energy,hits", [(0, 0), (3, 3), (4, 8), (5, 10)])
def test_heavenly_drill_spends_x_and_doubles_at_four(energy, hits):
    state = _fight(
        _source(
            "HEAVENLY_DRILL",
            "Deal 8 damage X times. Double X if it's 4 or more.",
            damage=8,
            cost=-1,
            is_x_cost=True,
        )
    )
    player = state.player.model_copy(update={"energy": energy})
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(
                update={"player": player},
            ),
        }
    )
    state = _play_first_card(state, "heavenly_drill")
    assert state.combat.monsters[0].hp == 100 - hits * 8
    assert state.player.energy == 0


@pytest.mark.parametrize("lost,hits", [(False, 1), (True, 2)])
def test_spite_checks_actual_hp_loss_this_turn(lost, hits):
    spec = _source(
        "SPITE", "Deal 5 damage. If you lost HP this turn, hits 2 times.", damage=5, hit_count=2
    )
    loss = _card("hurt", effects={"hp_loss": 1})
    state = _combat(_monster("target", {}), deck=(spec, loss))
    if lost:
        state = _play_first_card(state, "hurt")
    state = _play_first_card(state, "spite")
    assert state.combat.monsters[0].hp == 100 - hits * 5


def test_tear_asunder_counts_loss_events_instead_of_hp_amount():
    spec = _source(
        "TEAR_ASUNDER",
        "Deal 5 damage. Hits an additional time for each time you lost HP this combat.",
        damage=5,
    )
    state = _combat(_monster("target", {}), deck=(spec, _card("hurt", effects={"hp_loss": 3})))
    state = _play_first_card(state, "hurt")
    state = load_state(serialize(state))
    state = _play_first_card(state, "tear_asunder")
    assert state.combat.monsters[0].hp == 90


@pytest.mark.parametrize("count,intangible", [(8, 0), (9, 1)])
def test_eidolon_exhausts_all_remaining_cards_then_checks_actual_count(count, intangible):
    state = _fight(
        _source(
            "EIDOLON",
            "Exhaust your Hand. If 9 cards were Exhausted this way, gain 1 Intangible.",
            type="Skill",
            target="self",
        )
    )
    fillers = tuple(
        CardInstance(instance_id=f"filler:{i}", card_id="defend", type="skill", cost=0)
        for i in range(count)
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": state.combat.hand + fillers,
                }
            )
        }
    )
    state = _play_first_card(state, "eidolon")
    assert len(state.combat.exhaust_pile) == count
    assert not state.combat.hand
    assert state.player.statuses.get("intangible", 0) == intangible
    assert not any(event.kind == "effect_stubbed" for event in state.replay_log[-1].events)


def test_blur_keeps_block_for_one_next_turn_only():
    state = _fight(
        _source(
            "BLUR",
            "Gain 5 Block. Block is not removed at the start of your next turn.",
            block=5,
            type="Skill",
            target="self",
        )
    )
    state = _play_first_card(state, "blur")
    state = step(load_state(serialize(state)), _end_turn_action(state))
    assert state.player.block == 2  # Three attack damage was absorbed.
    assert state.player.statuses.get("blur", 0) == 0
    state = step(state, _end_turn_action(state))
    assert state.player.block == 0


@pytest.mark.parametrize("card_id", ["BOLAS", "THRUMMING_HATCHET"])
def test_returning_attack_keeps_physical_identity_next_turn(card_id):
    state = _fight(
        _source(
            card_id,
            "Deal 3 damage. At the start of your next turn, return this to your Hand.",
            damage=3,
        )
    )
    physical = state.combat.hand[0].instance_id
    state = _play_first_card(state, card_id.lower())
    state = step(load_state(serialize(state)), _end_turn_action(state))
    assert any(card.instance_id == physical for card in state.combat.hand)
    assert not state.combat.metadata.get("return_cards_next_turn")
    all_cards = (
        *state.combat.hand,
        *state.combat.draw_pile,
        *state.combat.discard_pile,
        *state.combat.exhaust_pile,
    )
    assert sum(card.instance_id == physical for card in all_cards) == 1


def test_gambit_kills_only_on_actual_unblocked_attack_damage():
    state = _fight(
        _source(
            "THE_GAMBIT",
            "Gain 50 Block. If you take unblocked attack damage this combat, die.",
            block=50,
            type="Skill",
            target="self",
        )
    )
    state = _play_first_card(state, "the_gambit")
    state = step(state, _end_turn_action(state))
    assert state.player.hp > 0
    state = step(load_state(serialize(state)), _end_turn_action(state))
    assert state.player.hp == 0


def test_multi_hit_card_profile_preserves_damage_and_scaling_signal():
    from sts2sim.mechanics.semantics import card_mechanic_profile
    card = _source("HEAVENLY_DRILL", "Deal 8 damage X times. Double X if it's 4 or more.",
                   damage=8, cost=-1, is_x_cost=True)
    profile = card_mechanic_profile(card)
    assert profile.values["damage"] == 8
    assert "multi_hit" in profile.tags
    assert "scales_with:energy_spent" in profile.tags
