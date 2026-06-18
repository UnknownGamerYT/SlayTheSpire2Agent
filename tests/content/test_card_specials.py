from __future__ import annotations

from sts2sim.mechanics.card_specials import (
    card_special_blockers,
    card_special_events,
    card_special_plan,
    normalize_card_special_steps,
)


def _soul_payload(*, upgraded: bool = False) -> dict:
    draw = 3 if upgraded else 2
    return {
        "card": {
            "id": "SOUL",
            "name": "Soul+" if upgraded else "Soul",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "draw": draw,
            "description": f"Draw {draw} cards.",
            "keywords_key": ("Exhaust",),
            "exhausts": True,
            "upgraded": upgraded,
        },
        "temporary": True,
    }


def test_channel_and_evoke_text_become_executable_orb_markers() -> None:
    plan = card_special_plan(
        {
            "id": "ORB_ROUTINE",
            "description": (
                "[gold]Channel[/gold] 1 [gold]Lightning[/gold].\n"
                "[gold]Evoke[/gold] your rightmost Orb twice."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"channel_orb": {"orb": "lightning", "amount": 1}},
        {"evoke_orb": {"selector": "rightmost", "amount": 2}},
    )
    assert [event["metadata"]["classification"] for event in plan.events] == [
        "executable",
        "executable",
    ]


def test_if_kill_resource_gain_is_an_explicit_blocker_not_immediate_energy() -> None:
    plan = card_special_plan(
        {
            "id": "FINISHING_BLOW",
            "description": "Deal 24 damage.\nIf this kills an enemy, gain [energy:3].",
        }
    )

    assert plan.status == "explicit_blocker"
    assert plan.steps == ()
    assert plan.blockers == (
        {
            "explicit_blocker": {
                "kind": "if_kill_resource_gain",
                "reason": "requires post-damage kill detection before granting the resource",
                "metadata": {
                    "resource": "energy",
                    "amount": 3,
                    "trigger": "card_kills_enemy",
                },
            }
        },
    )
    assert plan.events[0]["metadata"]["classification"] == "explicit_blocker"


def test_summon_osty_and_soul_text_emit_character_markers() -> None:
    card_spec = {
        "id": "NECRO_ASSIST",
        "description": (
            "[gold]Summon[/gold] 4.\n"
            "[gold]Osty[/gold] deals 6 damage to a random enemy.\n"
            "Add 2 [gold]Souls[/gold] into your [gold]Draw Pile[/gold]."
        ),
    }
    steps = normalize_card_special_steps(card_spec)

    assert steps == (
        {"player_resource": {"resource": "summon", "amount": 4, "source": "card_special"}},
        {"osty_action": {"action": "damage", "amount": 6, "target": "random_enemy"}},
        {"add_card_to_draw": (_soul_payload(), _soul_payload())},
    )
    assert card_special_blockers(card_spec) == ()


def test_stance_and_mantra_text_emit_executable_status_and_resource_steps() -> None:
    plan = card_special_plan(
        {
            "id": "INNER_FIRE",
            "description": "Enter Wrath.\nGain 4 Mantra.\nExit your Stance.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"apply_status": {"target": "self", "stance_wrath": 1}},
        {"player_resource": {"resource": "mantra", "amount": 4, "source": "card_special"}},
        {"apply_status": {"target": "self", "stance_none": 1}},
    )


def test_forge_executes_but_sovereign_blade_text_is_blocked() -> None:
    plan = card_special_plan(
        {
            "id": "BLADE_ORDER",
            "description": (
                "[gold]Forge[/gold] 3.\n"
                "[gold]Sovereign Blade[/gold] deals double damage to the enemy this turn.\n"
                "[gold]Sovereign Blade[/gold] now hits an additional time."
            ),
        }
    )

    assert plan.status == "partial"
    assert plan.steps == (
        {"player_resource": {"resource": "forge", "amount": 3, "source": "card_special"}},
    )
    assert [blocker["explicit_blocker"]["kind"] for blocker in plan.blockers] == [
        "sovereign_blade_state_required",
        "sovereign_blade_state_required",
    ]


def test_choose_card_required_text_is_classified_as_blocker_with_reason() -> None:
    blockers = card_special_blockers(
        {
            "id": "MINION_SWAP",
            "description": (
                "Choose 2 cards in your [gold]Draw Pile[/gold] to [gold]Transform[/gold] "
                "into [gold]Minion Dive Bombs[/gold]."
            ),
        }
    )

    assert blockers == (
        {
            "explicit_blocker": {
                "kind": "choose_card_required",
                "reason": (
                    "requires choosing card targets from a combat zone before the card can resolve"
                ),
                "metadata": {
                    "text": (
                        "choose 2 cards in your draw pile to transform into minion dive bombs"
                    ),
                    "zone": "draw_pile",
                    "count": 2,
                    "choices": None,
                    "action": "transform",
                },
            }
        },
    )


def test_dynamic_orb_text_blocks_without_losing_static_orb_slot_marker() -> None:
    plan = card_special_plan(
        {
            "id": "ORB_TRADE",
            "description": (
                "Lose 1 Orb Slot.\n"
                "[gold]Channel[/gold] 1 [gold]Frost[/gold] for each enemy."
            ),
        }
    )

    assert plan.status == "partial"
    assert plan.steps == ({"orb_slot_delta": {"amount": -1}},)
    assert plan.blockers[0]["explicit_blocker"]["kind"] == "dynamic_orb_channel_count"


def test_card_special_events_helper_returns_same_event_shape() -> None:
    events = card_special_events(
        {
            "id": "SOUL_CALL",
            "description": "Add a [gold]Soul[/gold] into your [gold]Hand[/gold].",
        }
    )

    assert events == (
        {
            "kind": "card_special_normalized",
            "source_id": "soul_call",
            "target_id": None,
            "amount": 1,
            "metadata": {
                "classification": "executable",
                "special": "add_soul",
                "effect": {
                    "add_card_to_hand": (_soul_payload(),),
                },
            },
        },
    )
