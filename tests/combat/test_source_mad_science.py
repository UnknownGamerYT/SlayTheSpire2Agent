"""All nine saved Mad Science rider variants execute through public plays."""

import pytest
from combat.test_source_modifiers import _card, _fight, _play
from combat.test_stored_turn_powers import _source_card

from sts2sim import load_state, serialize
from sts2sim.engine.transitions import (
    _card_has_keyword,
    _downgrade_card_instance,
    _energy_cost,
    _tinker_custom_card,
)
from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind

RIDERS = (
    ("attack", "sapping"),
    ("attack", "violence"),
    ("attack", "choking"),
    ("skill", "energized"),
    ("skill", "wisdom"),
    ("skill", "chaos"),
    ("power", "expertise"),
    ("power", "curious"),
    ("power", "improvement"),
)


def _mad(kind, rider, counter=1):
    return _tinker_custom_card(
        EventFlowMarker(
            kind=EventFlowMarkerKind.CUSTOM_CARD,
            qualifier=kind,
            metadata={"source_event_id": "tinker_time", "card_type": kind, "rider_id": rider},
        ),
        counter,
    )


@pytest.mark.parametrize("kind,rider", RIDERS)
def test_every_mad_science_variant_keeps_saved_native_type_and_upgrades_to_innate(kind, rider):
    mad = _mad(kind, rider)
    assert mad.card_id == "mad_science"
    assert mad.type.value == kind
    state = _fight(_source_card("armaments", upgraded=True), mad)
    result = _play(load_state(serialize(state)), "armaments")
    upgraded = result.combat.hand[0]
    assert upgraded.upgraded and _card_has_keyword(upgraded, "innate")
    assert upgraded.type.value == kind
    assert upgraded.custom["rider_id"] == rider
    assert upgraded.model_dump(mode="json")["effects"] == mad.model_dump(mode="json")["effects"]
    downgraded = _downgrade_card_instance(upgraded)
    assert not downgraded.upgraded and not _card_has_keyword(downgraded, "innate")
    assert downgraded.model_dump(mode="json")["effects"] == mad.model_dump(mode="json")["effects"]


@pytest.mark.parametrize("rider,hits", [("sapping", 1), ("violence", 3), ("choking", 1)])
def test_attack_riders_apply_strength_per_hit_and_choking_does_not_trigger_on_registration(
    rider, hits
):
    mad = _mad("attack", rider)
    state = _fight(mad, statuses={"strength": 2})
    result = _play(load_state(serialize(state)), mad.instance_id)
    assert result.combat.monsters[0].hp == 500 - hits * 14
    if rider == "sapping":
        assert result.combat.monsters[0].statuses == {"weak": 2, "vulnerable": 2}
    if rider == "choking":
        assert result.combat.monsters[0].statuses["strangle"] == 6


def test_choking_binds_the_original_target_and_triggers_on_later_card_plays():
    mad = _mad("attack", "choking")
    state = _fight(mad, _card("later"))
    result = _play(_play(state, mad.instance_id), "later")
    assert result.combat.monsters[0].hp == 482


def test_sapping_applies_weak_before_vulnerable_against_artifact():
    mad = _mad("attack", "sapping")
    state = _fight(mad, enemy_statuses={"artifact": 1})
    result = _play(state, mad.instance_id)
    assert result.combat.monsters[0].statuses == {"vulnerable": 2}


@pytest.mark.parametrize("rider", ["energized", "wisdom", "chaos"])
def test_skill_riders_gain_block_then_execute_native_rider(rider):
    mad = _mad("skill", rider)
    state = _fight(mad, draw=tuple(_card(f"drawn:{i}") for i in range(4)))
    if rider == "chaos":
        state = state.model_copy(update={"character_id": "DEFECT"})
    result = _play(load_state(serialize(state)), mad.instance_id)
    assert result.player.block == 8
    if rider == "energized":
        assert result.player.energy == 21
    elif rider == "wisdom":
        assert len(result.combat.hand) == 3
    else:
        assert len(result.combat.hand) == 1
        generated = result.combat.hand[0]
        assert generated.custom["free_to_play_this_turn"]
        assert _energy_cost(generated, result.player.energy, combat=result.combat) == 0


def test_expertise_applies_both_native_stats_and_curious_changes_later_power_cost():
    expertise = _mad("power", "expertise")
    result = _play(_fight(expertise), expertise.instance_id)
    assert result.player.statuses == {"strength": 2, "dexterity": 2}
    curious = _mad("power", "curious")
    state = _fight(curious, _card("later", "power", cost=2))
    result = _play(state, curious.instance_id)
    assert result.player.statuses["curious"] == 1
    assert _energy_cost(result.combat.hand[0], 20, combat=result.combat) == 1


def test_improvement_upgrades_distinct_eligible_deck_cards_including_eternal_at_combat_end():
    improvement = _mad("power", "improvement")
    state = _fight(
        improvement, _card("kill", "attack", effects={"damage": 999}), statuses={"improvement": 1}
    )
    deck = (
        _card("eternal", eternal=True),
        _card("normal"),
        _source_card("burn"),
        _card("upgraded").model_copy(update={"upgraded": True}),
    )
    from sts2sim.engine.rng import RunRng, capture_random_state

    state = state.model_copy(update={
        "master_deck": deck,
        "rng": capture_random_state(RunRng(
            "IMPROVEMENT", derivation="sts2.v0.107.1.utf16-djb2-splitmix64",
        )),
    })
    state = _play(load_state(serialize(state)), improvement.instance_id)
    assert not state.master_deck[0].upgraded
    result = _play(load_state(serialize(state)), "kill")
    assert result.master_deck[0].upgraded
    assert result.master_deck[1].upgraded
    assert not result.master_deck[2].upgraded
    assert result.rng.streams["combat_card_selection"].draws == 2
