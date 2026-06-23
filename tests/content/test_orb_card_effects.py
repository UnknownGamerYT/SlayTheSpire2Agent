from __future__ import annotations

from sts2sim.mechanics.orb_card_effects import (
    channel_orb_effect,
    chill_channel_count,
    compile_driver_draw,
    consuming_shadow_trigger,
    coolant_block,
    darkness_passive_trigger,
    lightning_rod_trigger,
    loop_passive_trigger,
    metadata_orb_channel_count,
    open_orb_slots,
    orb_slot_channel_count,
    passive_trigger_targets,
    spinner_effects,
    storm_trigger,
    synchronize_focus,
    tempest_channel_count,
    thunder_trigger,
    trash_to_treasure_trigger,
    unique_orb_types,
    voltaic_channel_count,
)


def test_unique_orb_scalers_use_current_orb_types() -> None:
    orbs = (
        {"orb_id": "Lightning"},
        {"orb": "frost"},
        {"id": "DARK", "value": 12},
        {"orb_id": "lightning"},
    )

    assert unique_orb_types(orbs) == ("lightning", "frost", "dark")
    assert compile_driver_draw(orbs) == 3
    assert coolant_block(orbs) == 6
    assert synchronize_focus(orbs) == 6


def test_dynamic_channel_counts_from_enemies_energy_slots_and_history() -> None:
    enemies = (
        {"monster_id": "alive_by_hp", "hp": 12},
        {"monster_id": "dead_by_hp", "hp": 0},
        {"monster_id": "alive_flag", "alive": True},
        {"monster_id": "dead_flag", "alive": False},
    )
    events = (
        {"kind": "orb_channeled", "amount": 1, "metadata": {"orb": "lightning"}},
        {"kind": "orb_channeled", "metadata": {"orb": "frost"}},
        {"kind": "orb_channeled", "amount": 3, "metadata": {"orb": "Lightning"}},
    )

    assert chill_channel_count(enemies) == 2
    assert tempest_channel_count(energy_spent=3) == 3
    assert tempest_channel_count(energy_spent=3, upgraded=True) == 4
    assert open_orb_slots(({"orb": "lightning"},), orb_slots=3) == 2
    assert orb_slot_channel_count(3) == 3
    assert orb_slot_channel_count(3, orbs=({"orb": "lightning"},), open_slots_only=True) == 2
    assert voltaic_channel_count(events=events) == 4


def test_voltaic_channel_count_can_use_metadata_or_explicit_count() -> None:
    metadata = {
        "orb_channel_counts": {
            "Lightning": 5,
            "frost": 2,
        }
    }

    assert metadata_orb_channel_count(metadata, orb_id="lightning") == 5
    assert voltaic_channel_count(metadata=metadata) == 5
    assert voltaic_channel_count(channel_count=7, metadata=metadata) == 7


def test_named_timed_triggers_emit_combat_trigger_payloads() -> None:
    assert storm_trigger(2).as_effect() == {
        "combat_trigger": {
            "trigger": "card_played",
            "duration": "combat",
            "effects": ({"channel_orb": {"orb": "lightning", "amount": 2}},),
            "condition": {"card_type": "power"},
        }
    }
    assert lightning_rod_trigger(turns=2).as_effect() == {
        "combat_trigger": {
            "trigger": "turn_start",
            "duration": "uses",
            "effects": ({"channel_orb": {"orb": "lightning", "amount": 1}},),
            "uses": 2,
        }
    }
    assert consuming_shadow_trigger().as_effect() == {
        "combat_trigger": {
            "trigger": "turn_end",
            "duration": "combat",
            "effects": ({"evoke_orb": {"selector": "leftmost", "amount": 1}},),
        }
    }
    assert trash_to_treasure_trigger(2).as_effect() == {
        "combat_trigger": {
            "trigger": "status_created",
            "duration": "combat",
            "effects": ({"channel_orb": {"orb": "random_orb", "amount": 2}},),
        }
    }


def test_spinner_and_thunder_named_helpers_cover_special_power_shapes() -> None:
    assert spinner_effects(upgraded=True) == (
        {"channel_orb": {"orb": "glass", "amount": 1}},
        {
            "combat_trigger": {
                "trigger": "turn_start",
                "duration": "combat",
                "effects": ({"channel_orb": {"orb": "glass", "amount": 1}},),
            }
        },
    )
    assert thunder_trigger(8).as_effect() == {
        "combat_trigger": {
            "trigger": "orb_evoked",
            "duration": "combat",
            "effects": (
                {"orb_evoke_damage": {"amount": 8, "target": "enemies_hit"}},
            ),
            "condition": {"orb": "lightning"},
        }
    }


def test_passive_trigger_descriptors_and_target_selection() -> None:
    orbs = (
        {"orb": "lightning"},
        {"orb": "dark"},
        {"orb": "frost"},
        {"orb": "dark"},
    )

    assert loop_passive_trigger(times=2).as_effect() == {
        "trigger_orb_passive": {"selector": "rightmost", "amount": 2}
    }
    assert loop_passive_trigger(times=2).as_repeated_effects() == (
        {"trigger_orb_passive": {"selector": "rightmost", "amount": 1}},
        {"trigger_orb_passive": {"selector": "rightmost", "amount": 1}},
    )
    assert darkness_passive_trigger().as_effect() == {
        "trigger_orb_passive": {
            "selector": "matching",
            "amount": 2,
            "orb": "dark",
            "direction": "left_to_right",
        }
    }
    assert passive_trigger_targets(orbs, selector="rightmost") == (3,)
    assert passive_trigger_targets(orbs, selector="leftmost", orb_id="dark") == (1,)
    assert passive_trigger_targets(orbs, selector="matching", orb_id="dark") == (1, 3)
    assert passive_trigger_targets(
        orbs,
        selector="matching",
        orb_id="dark",
        direction="right_to_left",
    ) == (3, 1)


def test_channel_effect_accepts_dynamic_amount_labels() -> None:
    assert channel_orb_effect("random", "orb_slots") == {
        "channel_orb": {"orb": "random_orb", "amount": "orb_slots"}
    }
