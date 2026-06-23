from __future__ import annotations

from sts2sim.mechanics.forge_blade import (
    ForgeContext,
    ForgeState,
    ForgeTrigger,
    SovereignBladeTarget,
    apply_conqueror,
    apply_forge,
    apply_parry,
    apply_seeking_edge,
    apply_sword_sage,
    beat_into_shape_forge_amount,
    conqueror_forge_amount,
    create_sovereign_blade_state,
    furnace_forge_descriptor,
    hammer_time_ally_forge_events,
    parry_block_amount,
    replace_sovereign_blade_in_zones,
    reset_forge_turn_counters,
    resolve_dynamic_forge_amount,
    resolve_forge_descriptor,
    resolve_parry_on_blade_play,
    seeking_edge_forge_amount,
    sovereign_blade_card,
    sovereign_blade_damage,
    sovereign_blade_from_card,
    sovereign_blade_hit_sequence,
    summon_forth_forge_amount,
    summon_forth_zones,
    tick_sovereign_blade_turn,
)


def test_forge_state_applies_resource_gain_and_turn_counters() -> None:
    amount = beat_into_shape_forge_amount(previous_hits_on_target_this_turn=3)
    result = apply_forge(
        amount,
        state=ForgeState(amount=2, gained_this_combat=4, times_forged_this_combat=1),
        resources={"forge": 2, "star": 1},
        source_id="beat_into_shape",
    )

    assert amount == 20
    assert result.resource_delta == 20
    assert result.resources == {"forge": 22, "star": 1}
    assert result.state.amount == 22
    assert result.state.gained_this_turn == 20
    assert result.state.gained_this_combat == 24
    assert result.state.times_forged_this_turn == 1
    assert result.state.times_forged_this_combat == 2
    assert result.events[0].source_id == "beat_into_shape"

    next_turn = reset_forge_turn_counters(result.state)
    assert next_turn.amount == 22
    assert next_turn.gained_this_turn == 0
    assert next_turn.times_forged_this_combat == 2


def test_dynamic_forge_formulas_cover_beat_into_shape_and_x_cost() -> None:
    upgraded_context = ForgeContext(previous_hits_on_target_this_turn=2, upgraded=True)
    x_context = ForgeContext(energy_spent=3)

    assert resolve_dynamic_forge_amount("beat_into_shape", upgraded_context) == 21
    assert resolve_dynamic_forge_amount("x_spent", x_context) == 3


def test_furnace_and_hammer_time_descriptors_emit_mapping_friendly_events() -> None:
    furnace = furnace_forge_descriptor(upgraded=True)
    furnace_events = resolve_forge_descriptor(furnace, ForgeTrigger.TURN_START)

    assert furnace.amount == 6
    assert furnace.repeat is True
    assert furnace_events[0].amount == 6
    assert furnace_events[0].target_id == "player"
    assert furnace_events[0].metadata["duration"] == "combat"
    assert resolve_forge_descriptor(furnace, ForgeTrigger.CARD_PLAYED) == ()

    hammer_events = hammer_time_ally_forge_events(9, ("osty", "ally_2"))

    assert [(event.target_id, event.amount) for event in hammer_events] == [
        ("osty", 9),
        ("ally_2", 9),
    ]
    assert all(event.source_id == "hammer_time" for event in hammer_events)


def test_known_regent_forge_and_parry_amount_helpers() -> None:
    assert conqueror_forge_amount() == 3
    assert conqueror_forge_amount(upgraded=True) == 5
    assert seeking_edge_forge_amount() == 7
    assert seeking_edge_forge_amount(upgraded=True) == 11
    assert summon_forth_forge_amount() == 8
    assert summon_forth_forge_amount(upgraded=True) == 11
    assert parry_block_amount() == 10
    assert parry_block_amount(upgraded=True) == 14


def test_sovereign_blade_operations_modify_state_without_engine_models() -> None:
    blade = create_sovereign_blade_state(instance_id="blade-1")
    blade = apply_parry(apply_seeking_edge(apply_sword_sage(blade)), amount=10)
    blade = apply_conqueror(blade, target_id="cultist")

    assert blade.target is SovereignBladeTarget.ALL_ENEMIES
    assert blade.hits == 1
    assert blade.replay == 1
    assert blade.block == 10
    assert sovereign_blade_damage(blade, target_id="cultist") == 20
    assert sovereign_blade_damage(blade, target_id="jaw_worm") == 10
    assert sovereign_blade_hit_sequence(blade, target_id="cultist") == (20,)

    expired = tick_sovereign_blade_turn(blade)

    assert expired.conqueror_marks == ()
    assert sovereign_blade_damage(expired, target_id="cultist") == 10


def test_sovereign_blade_card_mapping_round_trips_modified_state() -> None:
    blade = apply_parry(
        apply_sword_sage(
            apply_seeking_edge(create_sovereign_blade_state(instance_id="blade-2", zone="draw"))
        ),
        amount=14,
    )

    card = sovereign_blade_card(blade)
    recovered = sovereign_blade_from_card(card, zone="draw_pile")

    assert card["target"] == "all_enemies"
    assert card["effects"] == {"sequence": [{"all_damage": 10}, {"block": 14}]}
    assert card["custom"]["replay"] == 1
    assert recovered.instance_id == "blade-2"
    assert recovered.zone == "draw_pile"
    assert recovered.hits == 1
    assert recovered.block == 14
    assert recovered.replay == 1
    assert recovered.target is SovereignBladeTarget.ALL_ENEMIES


def test_summon_forth_moves_existing_blade_to_hand_or_creates_it() -> None:
    blade_card = sovereign_blade_card(
        create_sovereign_blade_state(instance_id="blade-3", zone="draw_pile")
    )
    zones = {
        "draw_pile": ({"card_id": "strike"}, blade_card),
        "discard_pile": ({"card_id": "defend"},),
        "hand": (),
    }

    moved = summon_forth_zones(zones)

    assert moved.created is False
    assert moved.previous_zone == "draw_pile"
    assert [card["card_id"] for card in moved.zones["draw_pile"]] == ["strike"]
    assert moved.zones["hand"][-1]["card_id"] == "sovereign_blade"
    assert moved.blade.zone == "hand"
    assert moved.events[0].kind == "sovereign_blade_moved"

    created = summon_forth_zones({"hand": ()})

    assert created.created is True
    assert created.zones["hand"][0]["card_id"] == "sovereign_blade"


def test_replace_sovereign_blade_in_zones_updates_existing_card_mapping() -> None:
    zones = {"discard_pile": (sovereign_blade_card(),)}
    blade = apply_parry(
        apply_sword_sage(create_sovereign_blade_state(zone="discard_pile"), amount=2),
        amount=10,
    )

    result = replace_sovereign_blade_in_zones(zones, blade)

    card = result.zones["discard_pile"][0]
    assert card["hit_count"] == 1
    assert card["block"] == 10
    assert card["custom"]["replay"] == 2
    assert card["effects"] == {"sequence": [{"damage": 10}, {"block": 10}]}
    assert result.blade.hits == 1
    assert result.blade.replay == 2
    assert result.blade.block == 10


def test_parry_only_triggers_for_sovereign_blade() -> None:
    parry = resolve_parry_on_blade_play({"card_id": "SOVEREIGN_BLADE"}, upgraded=True)
    miss = resolve_parry_on_blade_play({"card_id": "strike"}, upgraded=True)

    assert parry.block == 14
    assert parry.events[0].kind == "gain_block"
    assert miss.block == 0
    assert miss.events == ()
