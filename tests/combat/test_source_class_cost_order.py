"""Class costs participate in native local/normal-global/late-global order."""

import pytest
from combat.test_source_modifiers import _card, _fight, _play, _plays
from combat.test_stored_turn_powers import _source_card

from sts2sim import load_state, serialize
from sts2sim.engine.source_cost_hooks import set_local_energy_cost
from sts2sim.engine.transitions import _card_has_keyword, _energy_cost


def _free(state, identity):
    player = state.player.model_copy(update={"energy": 0})
    return state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(
                update={
                    "player": player,
                    "metadata": {
                        **state.combat.metadata,
                        "source_class_cost_powers": {identity: {"amount": 1}},
                    },
                }
            ),
        }
    )


def test_free_power_late_modifier_overrides_spiked_gauntlets_surcharge():
    state = _free(_fight(_card("power", "power", cost=3)), "free_power")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {
                        **state.combat.metadata,
                        "item_relic_ids": ("spiked_gauntlets",),
                    }
                }
            )
        }
    )
    assert "power" in _plays(state)
    result = _play(state, "power")
    assert result.player.energy == 0
    assert result.combat.metadata["source_class_cost_powers"]["free_power"]["amount"] == 0


@pytest.mark.parametrize("hexed", [False, True])
def test_free_ethereal_late_modifier_overrides_borrowed_time_and_sees_hex_keywords(hexed):
    state = _fight(
        _card("skill", cost=2, ethereal=not hexed),
        statuses={"borrowed_time": 1, **({"hex": 2} if hexed else {})},
    )
    state = _free(state, "free_ethereal")
    result = _play(state, "skill")
    assert result.player.energy == 0
    assert result.combat.metadata["source_class_cost_powers"]["free_ethereal"]["amount"] == 0


def test_hex_derived_ethereal_keyword_is_visible_to_other_card_hooks():
    state = _fight(_card("first"), _card("hexed"), statuses={"hex": 2})
    result = _play(state, "first")
    assert _card_has_keyword(result.combat.hand[0], "ethereal")
    player = result.player.model_copy(update={"statuses": {}})
    result = result.model_copy(
        update={"player": player, "combat": result.combat.model_copy(update={"player": player})}
    )
    result = _play(result, "hexed")
    assert not _card_has_keyword(result.combat.discard_pile[-1], "ethereal")


def test_curious_power_reduces_only_positive_power_costs():
    state = _fight(_card("power", "power", cost=2), _card("skill", cost=2), statuses={"curious": 1})
    assert _energy_cost(state.combat.hand[0], 20, combat=state.combat) == 1
    assert _energy_cost(state.combat.hand[1], 20, combat=state.combat) == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_rocket_punch_discount_expiry_preserves_later_absolute_cost_modifiers(legacy):
    rocket = _source_card("rocket_punch")
    if legacy:
        rocket = rocket.model_copy(
            update={
                "cost": 0,
                "custom": {
                    **rocket.custom,
                    "rocket_punch_cost_before_discount": 2,
                },
            }
        )
    else:
        rocket = set_local_energy_cost(rocket, 0, "played")
    rocket = set_local_energy_cost(rocket, 1, "combat")
    result = _play(load_state(serialize(_fight(rocket))), "rocket_punch")
    assert result.player.energy == 19
    assert result.combat.discard_pile[0].cost == 1
    assert result.combat.discard_pile[0].custom["source_local_energy_base"] == 2


def test_adaptive_strike_clone_uses_a_combat_modifier_preserving_its_base():
    result = _play(_fight(_source_card("adaptive_strike")), "adaptive_strike")
    clone = next(c for c in result.combat.discard_pile if c.instance_id != "adaptive_strike")
    assert clone.cost == 0
    assert clone.custom["source_local_energy_base"] == 2
