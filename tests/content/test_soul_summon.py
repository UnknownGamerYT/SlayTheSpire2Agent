from __future__ import annotations

from sts2sim.engine import CardInstance, CardType, CombatState, PlayerState, TargetType
from sts2sim.mechanics.soul_summon import (
    SoulPileCounts,
    SoulPileZone,
    add_soul_effect_step,
    count_souls,
    devour_life_soul_play_trigger,
    dynamic_soul_creation_count,
    dynamic_summon_amount,
    haunt_soul_play_trigger,
    is_soul_card,
    next_turn_summon_trigger,
    resolve_dirge_x_cost,
    resolve_soul_count_scaling,
    soul_card_payload,
    soul_count_scaled_amount,
    soul_pile_counts,
    soul_pile_counts_from_combat,
    soul_pile_zone,
    summon_effect_step,
    timed_summon_trigger,
)


def _soul_instance(instance_id: str, *, upgraded: bool = False) -> CardInstance:
    return CardInstance(
        instance_id=instance_id,
        card_id="soul",
        name="Soul+" if upgraded else "Soul",
        type=CardType.SKILL,
        target=TargetType.SELF,
        cost=1,
        exhausts=True,
        upgraded=upgraded,
    )


def test_soul_card_detection_is_conservative() -> None:
    assert is_soul_card(_soul_instance("s1"))
    assert is_soul_card({"id": "SOUL", "name": "Soul+"})
    assert is_soul_card({"card": {"id": "SOUL", "name": "Soul"}})
    assert is_soul_card("SOUL")

    assert not is_soul_card({"id": "SOUL_STORM", "name": "Soul Storm"})
    assert not is_soul_card({"id": "FORGOTTEN_SOUL", "name": "Forgotten Soul"})


def test_soul_counts_cover_all_combat_piles_and_zone_aliases() -> None:
    counts = soul_pile_counts(
        hand=(_soul_instance("hand"), {"id": "STRIKE"}),
        draw_pile=({"card_id": "SOUL"}, {"id": "SOUL_STORM"}),
        discard_pile=("SOUL",),
        exhaust_pile=({"card": {"id": "SOUL"}}, {"id": "DEFEND"}),
    )

    assert counts == SoulPileCounts(hand=1, draw_pile=1, discard_pile=1, exhaust_pile=1)
    assert counts.total == 4
    assert counts.for_zone("draw") == 1
    assert counts.for_zone(SoulPileZone.EXHAUST_PILE) == 1
    assert counts.for_zone("all") == 4
    assert counts.as_mapping() == {
        "hand": 1,
        "draw_pile": 1,
        "discard_pile": 1,
        "exhaust_pile": 1,
        "total": 4,
    }
    assert count_souls((_soul_instance("a"), "SOUL", {"id": "SOUL_STORM"})) == 2
    assert soul_pile_zone("exhaust") is SoulPileZone.EXHAUST_PILE


def test_soul_counts_accept_combat_state_or_mapping() -> None:
    combat = CombatState(
        player=PlayerState(hp=70, max_hp=80),
        hand=(_soul_instance("hand"),),
        draw_pile=(_soul_instance("draw"),),
        discard_pile=(),
        exhaust_pile=(_soul_instance("exhaust"),),
    )

    assert soul_pile_counts_from_combat(combat) == SoulPileCounts(
        hand=1,
        draw_pile=1,
        discard_pile=0,
        exhaust_pile=1,
    )
    assert soul_pile_counts_from_combat(
        {
            "hand": ({"id": "SOUL"},),
            "draw": ({"id": "SOUL"},),
            "discard": (),
            "exhaust": ({"id": "SOUL"}, {"id": "BASH"}),
        }
    ).total == 3


def test_soul_count_scaling_resolves_bonus_and_total_amounts() -> None:
    counts = SoulPileCounts(hand=1, exhaust_pile=3)

    result = resolve_soul_count_scaling(
        counts,
        zone="exhaust_pile",
        base_amount=9,
        amount_per_soul=2,
    )

    assert result.soul_count == 3
    assert result.bonus_amount == 6
    assert result.total_amount == 15
    assert result.as_mapping() == {
        "zone": "exhaust_pile",
        "soul_count": 3,
        "amount_per_soul": 2,
        "base_amount": 9,
        "bonus_amount": 6,
        "total_amount": 15,
    }
    assert soul_count_scaled_amount(
        counts,
        zone="all",
        base_amount=0,
        amount_per_soul=1,
    ) == 4


def test_soul_play_trigger_descriptors_cover_devour_life_and_haunt() -> None:
    devour = devour_life_soul_play_trigger(2)
    haunt = haunt_soul_play_trigger(6)

    assert devour.condition == {"card_id": "soul", "is_soul": True}
    assert devour.combat_trigger_payload()["effects"] == (
        {"player_resource": {"resource": "summon", "amount": 2, "source": "devour_life"}},
    )
    assert devour.to_mapping()["combat_trigger"]["metadata"] == {"style": "devour_life"}

    assert haunt.combat_trigger_payload()["condition"] == {"card_id": "soul", "is_soul": True}
    assert haunt.combat_trigger_payload()["effects"] == (
        {"enemy_hp_loss": {"target": "random_enemy", "amount": 6, "source": "haunt"}},
    )
    assert haunt.to_mapping()["combat_trigger"]["metadata"] == {"style": "haunt"}


def test_dynamic_summon_and_soul_creation_counts_for_x_cost_cards() -> None:
    summon = dynamic_summon_amount(energy_spent=3, amount_per_energy=4)
    souls = dynamic_soul_creation_count(energy_spent=3)
    dirge = resolve_dirge_x_cost(3, summon_per_energy=4, upgraded=True)

    assert summon.amount == 12
    assert summon.as_mapping()["kind"] == "dynamic_summon_amount"
    assert souls.amount == 3
    assert souls.as_mapping()["kind"] == "dynamic_soul_creation_count"

    assert dirge.summon_amount == 12
    assert dirge.soul_count == 3
    assert dirge.effect_steps()[0] == {
        "player_resource": {"resource": "summon", "amount": 12, "source": "dirge"}
    }
    generated_step = dirge.effect_steps()[1]
    assert tuple(generated_step) == ("add_card_to_draw",)
    generated = generated_step["add_card_to_draw"]
    assert isinstance(generated, tuple)
    assert len(generated) == 3
    assert generated[0] == soul_card_payload(upgraded=True)


def test_add_soul_and_summon_effect_steps_are_mapping_friendly() -> None:
    assert summon_effect_step(5, source="Bodyguard") == {
        "player_resource": {"resource": "summon", "amount": 5, "source": "bodyguard"}
    }
    assert add_soul_effect_step(2, destination="discard_pile") == {
        "add_card_to_discard": (
            soul_card_payload(upgraded=False),
            soul_card_payload(upgraded=False),
        )
    }


def test_timed_summon_trigger_descriptors_cover_next_turn_and_repeating_cases() -> None:
    next_turn = next_turn_summon_trigger(2)
    repeating = timed_summon_trigger(
        1,
        source_id="lich_glass",
        trigger="turn_start",
        duration="combat",
    )

    assert next_turn.combat_trigger_payload() == {
        "trigger": "turn_start",
        "duration": "once",
        "effects": (
            {
                "player_resource": {
                    "resource": "summon",
                    "amount": 2,
                    "source": "summon_next_turn",
                }
            },
        ),
        "source_id": "summon_next_turn",
        "source_card_id": "summon_next_turn",
        "text": "At the start of your next turn, Summon 2.",
        "metadata": {"style": "next_turn_summon"},
    }
    assert repeating.to_mapping() == {
        "combat_trigger": {
            "trigger": "turn_start",
            "duration": "combat",
            "effects": (
                {
                    "player_resource": {
                        "resource": "summon",
                        "amount": 1,
                        "source": "lich_glass",
                    }
                },
            ),
            "source_id": "lich_glass",
            "source_card_id": "lich_glass",
        }
    }
