from __future__ import annotations

from sts2sim.mechanics import (
    CombatRelicHook,
    card_played,
    combat_end,
    damage_dealt,
    damage_taken,
    start_of_combat,
    supported_combat_relic_ids,
    turn_end,
    turn_start,
    unsupported_combat_relic_handlers,
)


def test_start_of_combat_outputs_markers_and_capped_heal() -> None:
    result = start_of_combat(
        ("ANCHOR", {"name": "Blood Vial"}, "VAJRA"),
        player_hp=79,
        player_max_hp=80,
    )

    assert result.hook is CombatRelicHook.START_OF_COMBAT
    assert result.hp_delta == 1
    assert result.block_delta == 10
    assert result.blockers == ()
    assert [(marker.kind, marker.amount, marker.target_id) for marker in result.markers] == [
        ("gain_block", 10, "player"),
        ("heal_player", 1, "player"),
        ("gain_status", 1, "player"),
    ]
    assert result.markers[2].metadata == {"status": "strength"}
    assert result.markers[2].source_id == "vajra"


def test_start_of_combat_relics_emit_draw_vigor_and_focus_markers() -> None:
    result = start_of_combat(("AKABEKO", "BAG_OF_PREPARATION", "DATA_DISK"))

    assert [(marker.kind, marker.amount, marker.metadata) for marker in result.markers] == [
        ("gain_status", 8, {"status": "vigor"}),
        ("draw_cards", 2, {}),
        ("gain_status", 1, {"status": "focus"}),
    ]


def test_brimstone_turn_start_buffs_player_and_enemies() -> None:
    result = turn_start(("BRIMSTONE",))

    assert [(marker.kind, marker.amount, marker.target_id) for marker in result.markers] == [
        ("gain_status", 2, "player"),
        ("apply_status", 1, "all_enemies"),
    ]
    assert result.markers[0].metadata == {"status": "strength"}
    assert result.markers[1].metadata == {"status": "strength"}


def test_happy_flower_periodic_turn_start_energy_uses_turn_or_counter() -> None:
    quiet = turn_start(("HAPPY_FLOWER",), turn_number=2)
    triggered = turn_start(("HAPPY_FLOWER",), turn_number=3)
    counter_triggered = turn_start(("HAPPY_FLOWER",), relic_counters={"happy_flower": 2})

    assert quiet.markers == ()
    assert triggered.energy_delta == 1
    assert triggered.markers[0].kind == "gain_energy"
    assert triggered.markers[0].metadata == {"period": 3, "turn_number": 3}
    assert counter_triggered.energy_delta == 1
    assert counter_triggered.markers[0].metadata == {"period": 3, "next_counter": 0}


def test_card_played_attack_counter_relics_emit_only_on_threshold() -> None:
    missed = card_played(
        ("SHURIKEN",),
        card_type="attack",
        metadata={"attacks_played_this_turn": 1},
    )
    triggered = card_played(
        ("SHURIKEN", "KUNAI", "ORNAMENTAL_FAN"),
        card_type="attack",
        metadata={"attacks_played_this_turn": 2},
    )

    assert missed.markers == ()
    assert [(marker.kind, marker.amount) for marker in triggered.markers] == [
        ("gain_status", 1),
        ("gain_status", 1),
        ("gain_block", 4),
    ]
    assert triggered.markers[0].metadata["status"] == "strength"
    assert triggered.markers[1].metadata["status"] == "dexterity"
    assert triggered.block_delta == 4


def test_vulnerable_math_markers_for_odd_mushroom_and_paper_phrog() -> None:
    taken = damage_taken(("ODD_MUSHROOM",), player_statuses={"Vulnerable": 1})
    dealt = damage_dealt(("PAPER_PHROG",), target_statuses={"vulnerable": 1}, target_id="jaw_worm")

    assert taken.markers[0].kind == "modify_vulnerable_damage_taken"
    assert taken.markers[0].amount == 125
    assert taken.markers[0].metadata["normal_multiplier_percent"] == 150
    assert dealt.markers[0].kind == "modify_vulnerable_damage_dealt"
    assert dealt.markers[0].amount == 175
    assert dealt.markers[0].target_id == "jaw_worm"


def test_orichalcum_turn_end_and_preserved_insect_elite_markers_are_conditional() -> None:
    orichalcum = turn_end(("ORICHALCUM",), player_block=0)
    blocked = turn_end(("ORICHALCUM",), player_block=4)
    elite = start_of_combat(("PRESERVED_INSECT",), encounter_type="elite")
    normal = start_of_combat(("PRESERVED_INSECT",), encounter_type="normal")

    assert orichalcum.block_delta == 6
    assert orichalcum.markers[0].metadata == {
        "condition": "player_block_is_zero",
        "player_block": 0,
    }
    assert blocked.markers == ()
    assert elite.markers[0].kind == "elite_monster_hp_multiplier"
    assert elite.markers[0].amount == 75
    assert elite.markers[0].metadata["hp_reduction_percent"] == 25
    assert normal.markers == ()


def test_combat_end_healing_is_capped() -> None:
    result = combat_end(("BURNING_BLOOD", "BLACK_BLOOD"), player_hp=75, player_max_hp=80)

    assert result.hp_delta == 5
    assert [marker.amount for marker in result.markers] == [5, 0]


def test_unsupported_combat_relics_return_blockers_with_source_ids() -> None:
    relic = {
        "id": "UNHANDLED_STARTER",
        "name": "Unhandled Starter",
        "description": "At the start of each combat, do a very specific thing.",
    }

    result = start_of_combat((relic,))
    blockers = unsupported_combat_relic_handlers((relic,))

    assert len(result.blockers) == 1
    assert result.blockers[0].hook is CombatRelicHook.START_OF_COMBAT
    assert result.blockers[0].source_id == "unhandled_starter"
    assert "No pure combat relic helper" in result.blockers[0].reason
    assert blockers == result.blockers


def test_inventory_conditioned_relics_are_explicit_combat_blockers() -> None:
    relic = {
        "id": "BELT_BUCKLE",
        "name": "Belt Buckle",
        "description": (
            "While you have no potions, you have [blue]2[/blue] additional "
            "[gold]Dexterity[/gold]."
        ),
    }

    blockers = unsupported_combat_relic_handlers((relic,))

    assert len(blockers) == 1
    assert blockers[0].hook is CombatRelicHook.START_OF_COMBAT
    assert blockers[0].relic_id == "belt_buckle"


def test_supported_combat_relic_ids_are_hook_scoped() -> None:
    assert "anchor" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "bag_of_preparation" in supported_combat_relic_ids(
        CombatRelicHook.START_OF_COMBAT
    )
    assert "brimstone" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "happy_flower" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "orichalcum" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "unhandled_starter" not in supported_combat_relic_ids()
