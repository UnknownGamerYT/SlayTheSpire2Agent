"""Visible rich bridge fields import actual source state, not placeholder classes."""

from __future__ import annotations

from copy import deepcopy

import pytest
from live.test_live_ppo_policy import _combat

from sts2sim import load_state, serialize, step
from sts2sim.engine import Action
from sts2sim.engine.transitions import _energy_cost
from sts2sim.live_parity import map_live_action_to_simulator_action
from sts2sim.live_policy import LivePolicyUnsupported, import_policy_combat


def _card(identifier, *, cost=1, target="AnyEnemy", **fields):
    return {
        "id": identifier,
        "is_upgraded": False,
        "upgrade_level": 0,
        "index": 0,
        "can_play": True,
        "cost": str(cost),
        "target_type": target,
        **fields,
    }


def _with_card(card):
    state = _combat()
    state["player"]["hand"] = [card]
    state["player"]["master_deck"] = [deepcopy(card)]
    state["player"].pop("deck")
    return state


def test_source_enchantment_affliction_and_ordered_cost_rows_import_separately():
    raw = _with_card(
        _card(
            "BASH",
            cost=0,
            has_enchantment=True,
            enchantment={"id": "SHARP", "amount": 3},
            affliction={"id": "RINGING", "amount": 2},
            base_cost=2,
            local_cost=0,
            local_cost_modifiers=[
                {"amount": 0, "type": "Absolute", "expiration": 6, "reduce_only": False}
            ],
            base_star_cost=3,
            temporary_star_costs=[
                {"cost": 0, "clears_when_turn_ends": True, "clears_when_card_is_played": True}
            ],
        )
    )
    sim = import_policy_combat(raw)
    card = sim.combat.hand[0]
    assert card.enchantments[0].keyword == "sharp"
    assert card.enchantments[0].amount == 3
    assert card.custom["source_core_affliction"] == {"id": "ringing", "amount": 2}
    assert card.custom["source_local_energy_modifiers"][0]["expiration"] == 3
    assert card.custom["source_local_star_modifiers"][0]["expiration"] == 3
    assert _energy_cost(card, 3, combat=sim.combat) == 0
    restored = load_state(serialize(sim))
    assert serialize(restored) == serialize(sim)
    action = Action(type="play_card", card_instance_id=card.instance_id, target_id="NIBBIT_0")
    assert serialize(step(restored, action)) == serialize(step(sim, action))


def test_local_cost_is_not_replaced_by_effective_cost_and_charged_global_twice():
    raw = _with_card(_card("BASH", cost=4, base_cost=2, local_cost=2, local_cost_modifiers=[]))
    raw["player"]["status"] = [{"id": "BORROWED_TIME_POWER", "amount": 2}]
    raw["player"]["energy"] = 20
    sim = import_policy_combat(raw)
    assert sim.combat.hand[0].cost == 2
    assert _energy_cost(sim.combat.hand[0], 20, combat=sim.combat) == 4


def test_regent_stars_and_forged_blade_damage_repeats_are_observed():
    raw = _with_card(_card("SOVEREIGN_BLADE", cost=2, dynamic_vars={"Damage": 37, "Repeat": 2}))
    raw["player"].update(character_id="REGENT", stars=7, resources={"stars": 7})
    raw["battle"]["enemies"][0].update(hp=1000, max_hp=1000)
    sim = import_policy_combat(raw)
    assert sim.player.resources == {"star": 7}
    card = sim.combat.hand[0]
    result = step(
        sim, Action(type="play_card", card_instance_id=card.instance_id, target_id="NIBBIT_0")
    )
    assert result.combat.monsters[0].hp == 926


def test_defect_orb_accumulators_preserve_dark_and_unmodified_glass_values():
    raw = _combat()
    raw["player"].update(
        character_id="DEFECT",
        orb_slots=3,
        orbs=[
            {"id": "DARK_ORB", "evoke_val": 17},
            {"id": "GLASS_ORB", "base_value": 2, "passive_val": 5, "evoke_val": 10},
            {"id": "LIGHTNING_ORB", "passive_val": 6, "evoke_val": 11},
        ],
        status=[{"id": "FOCUS_POWER", "amount": 3}],
    )
    sim = import_policy_combat(raw)
    assert [(o.orb_id, o.value) for o in sim.combat.orbs] == [
        ("dark", 17),
        ("glass", 2),
        ("lightning", 0),
    ]
    bad = deepcopy(raw)
    bad["player"]["orbs"][1].pop("base_value")
    with pytest.raises(LivePolicyUnsupported, match="base_value"):
        import_policy_combat(bad)


def test_necrobinder_pet_retains_dead_identity_and_never_fabricates_alive_hp():
    raw = _combat()
    raw["player"].update(
        character_id="NECROBINDER",
        pets=[
            {
                "id": "OSTY",
                "alive": False,
                "hp": 0,
                "max_hp": 6,
                "block": 0,
                "status": [{"id": "DIE_FOR_YOU_POWER", "amount": 1}],
            }
        ],
    )
    sim = import_policy_combat(raw)
    assert sim.combat.metadata["osty"]["created"] is True
    assert sim.combat.metadata["osty"]["alive"] is False
    assert sim.combat.metadata["osty"]["hp"] == 0
    assert sim.combat.metadata["osty"]["max_hp"] == 6
    assert sim.combat.metadata["osty"]["die_for_you"] is True


def test_source_physical_identity_and_finished_history_translate_to_deck_identity():
    raw = _with_card(_card("BASH", cost=2, instance_id="combat:9", deck_instance_id="deck:2"))
    raw["player"]["master_deck"][0].update(instance_id="deck:2", deck_instance_id=None)
    raw["battle"].update(
        cards_played_this_turn=["combat:9", "combat:9"],
        combat_metadata={"attacks_played_this_turn": 2, "cards_played_this_combat": 5},
    )
    sim = import_policy_combat(raw)
    assert sim.master_deck[0].instance_id == sim.combat.hand[0].instance_id == "deck:2"
    assert sim.combat.cards_played_this_turn == ("deck:2", "deck:2")
    assert sim.combat.metadata["attacks_played_this_turn"] == 2


def test_wither_fake_upgrades_are_not_invalid_current_upgrade_levels():
    raw = _with_card(_card("WITHER", cost=0, target="None", fake_upgrade_level=3))
    raw["player"]["hand"][0]["can_play"] = False
    sim = import_policy_combat(raw)
    assert sim.combat.hand[0].custom["wither_upgrade_count"] == 3
    assert sim.combat.hand[0].upgraded is False
    bad = _with_card(_card("BASH", cost=2, upgrade_level=2, is_upgraded=True))
    with pytest.raises(LivePolicyUnsupported, match="Invalid source upgrade level"):
        import_policy_combat(bad)


def test_observed_banshee_local_discount_is_not_applied_again_from_history():
    raw = _with_card(
        _card(
            "BANSHEES_CRY",
            cost=5,
            target="AllEnemies",
            base_cost=9,
            local_cost=5,
            local_cost_modifiers=[
                {"amount": -4, "type": "Relative", "expiration": 0, "reduce_only": False}
            ],
        )
    )
    raw["player"]["energy"] = 20
    raw["battle"]["combat_metadata"] = {"source_class_ethereal_plays": 2}
    sim = import_policy_combat(raw)
    assert _energy_cost(sim.combat.hand[0], 20, combat=sim.combat) == 5


def test_one_play_exhaust_marker_is_consumed_without_permanent_keyword_mutation():
    raw = _with_card(_card("BASH", cost=2, exhaust_on_next_play=True))
    sim = import_policy_combat(raw)
    card = sim.combat.hand[0]
    result = step(
        sim, Action(type="play_card", card_instance_id=card.instance_id, target_id="NIBBIT_0")
    )
    assert len(result.combat.exhaust_pile) == 1
    exhausted = result.combat.exhaust_pile[0]
    assert not exhausted.exhausts
    assert not exhausted.custom.get("source_exhaust_on_next_play")


def test_live_physical_potion_slots_do_not_repack_or_alias_a_hole():
    raw = _combat()
    raw["player"]["potions"] = [{"id": "FIRE_POTION", "slot": 0}, {"id": "BLOCK_POTION", "slot": 2}]
    sim = import_policy_combat(raw)
    assert sim.flags["potion_slot_indices"] == (0, 2)
    action = map_live_action_to_simulator_action(raw, {"action": "use_potion", "slot": 2}, sim)
    assert action.payload["potion_id"] == "block_potion"
    assert action.payload["potion_slot"] == "potion:2"


def test_modifier_flag_without_payload_and_unknown_resource_fail_closed():
    raw = _with_card(_card("BASH", cost=2, has_enchantment=True))
    with pytest.raises(LivePolicyUnsupported, match="losslessly"):
        import_policy_combat(raw)
    raw = _combat()
    raw["player"]["resources"] = {"modded_resource": 3}
    with pytest.raises(LivePolicyUnsupported, match="Unknown live player resource"):
        import_policy_combat(raw)


def test_displayed_multiattack_uses_canonical_total_and_actual_unblocked_damage():
    raw = _combat()
    raw["battle"]["enemies"][0]["intents"] = [
        {"type": "Attack", "label": "4x2"},
        {"type": "Buff", "label": ""},
    ]
    sim = import_policy_combat(raw)
    assert sim.combat.monsters[0].intent_damage == 8
    assert sim.combat.monsters[0].hit_count == 2
    assert sim.combat.monsters[0].intent == "attack_buff"
    result = step(sim, Action(type="end_turn"))
    assert result.player.hp == 72


def test_player_turn_round_and_total_floor_are_distinct_observed_quantities():
    raw = _combat()
    raw["run"].update(act=2, floor=24, total_floor=24, act_floor=7)
    raw["battle"].update(round=3, player_turn=4)
    sim = import_policy_combat(raw)
    assert sim.floor == 7
    assert sim.flags["total_floor"] == 24
    assert sim.combat.turn == 4
    assert sim.combat.metadata["source_round_number"] == 3
    raw["run"].pop("act_floor")
    with pytest.raises(LivePolicyUnsupported, match="within-act floor"):
        import_policy_combat(raw)


def test_public_potion_commands_roundtrip_physical_slots_and_observed_permission():
    from sts2sim.live_policy import bridge_action_for_policy

    raw = _combat()
    raw["player"]["potions"] = [
        {"id": "BLOCK_POTION", "slot": 2, "can_use_in_combat": True, "target_type": "Self"}
    ]
    raw["player"]["can_remove_potions"] = True
    sim = import_policy_combat(raw)
    use = Action(
        type="use_potion",
        target_id="player",
        payload={"potion_slot": "potion:2", "potion_id": "block_potion"},
    )
    discard = Action(type="discard_potion", target_id="potion:2")
    assert bridge_action_for_policy(raw, sim, use) == {"action": "use_potion", "slot": 2}
    assert bridge_action_for_policy(raw, sim, discard) == {"action": "discard_potion", "slot": 2}
    raw["player"]["can_remove_potions"] = False
    assert bridge_action_for_policy(raw, sim, use) is None
    assert bridge_action_for_policy(raw, sim, discard) is None
    assert import_policy_combat(raw).flags["can_remove_potions"] is False


def test_finished_manual_history_does_not_count_autoplays_or_replay_indices_twice():
    raw = _combat()
    raw["battle"].update(
        player_turn=3,
        card_plays_this_turn=[
            {"is_auto_play": False, "play_index": 0},
            {"is_auto_play": False, "play_index": 1},
            {"is_auto_play": True, "play_index": 0},
        ],
    )
    sim = import_policy_combat(raw)
    assert sim.combat.metadata["card_status_turn_counters"] == {
        "turn": 3,
        "manual_cards": 1,
        "paels_eye_manual": 1,
        "brilliant_scarf_manual": 1,
    }


@pytest.mark.parametrize(
    ("character", "fields", "match"),
    [
        ("REGENT", {}, "explicit live stars"),
        ("DEFECT", {"orb_slots": 3}, "explicit live orbs"),
        ("NECROBINDER", {}, "explicit live pets"),
    ],
)
def test_missing_class_state_fails_before_policy_inference(character, fields, match):
    raw = _combat()
    raw["player"].update(character_id=character, **fields)
    with pytest.raises(LivePolicyUnsupported, match=match):
        import_policy_combat(raw)


def test_conflicting_star_aliases_and_heterogeneous_attack_hits_fail_closed():
    raw = _combat()
    raw["player"].update(character_id="REGENT", stars=7, resources={"star": 8})
    with pytest.raises(LivePolicyUnsupported, match="contradict"):
        import_policy_combat(raw)
    raw = _combat()
    raw["battle"]["enemies"][0]["intents"] = [
        {"type": "Attack", "label": "4x2"},
        {"type": "Attack", "label": "7"},
    ]
    with pytest.raises(LivePolicyUnsupported, match="hit sequence"):
        import_policy_combat(raw)
