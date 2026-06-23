from __future__ import annotations

from sts2sim.mechanics.reward_triggers import (
    RewardGenerationContext,
    RewardModifierSpec,
    resolve_reward_generation_triggers,
)
from sts2sim.mechanics.triggers import (
    GameTrigger,
    TriggerContext,
    TriggerDispatcher,
    TriggerEffect,
    TriggerHandler,
    TriggerHandlerResult,
)


def test_question_card_and_busted_crown_emit_card_choice_deltas() -> None:
    result = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="normal",
            card_choice_count=3,
            owned_relics=("QUESTION_CARD", "BUSTED-CROWN"),
        )
    )

    assert result.card_choice_delta == -1
    assert result.card_choice_count == 2
    assert [(effect.kind, effect.amount, effect.content_id) for effect in result.effects] == [
        ("reward_card_choice_delta", 1, "question_card"),
        ("reward_card_choice_delta", -2, "busted_crown"),
    ]
    assert result.effects[0].metadata["card_choice_count"] == 4
    assert result.effects[1].metadata["card_choice_count"] == 2
    assert result.trigger_resolution.trigger is GameTrigger.COMBAT_REWARD_GENERATED


def test_prayer_wheel_adds_extra_group_only_for_normal_combat_card_rewards() -> None:
    normal = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="normal",
            card_choice_count=3,
            owned_relics=("prayer_wheel",),
        )
    )
    elite = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="elite",
            card_choice_count=3,
            owned_relics=("prayer_wheel",),
        )
    )
    event = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="event",
            encounter_type="normal",
            card_choice_count=3,
            owned_relics=("prayer_wheel",),
        )
    )

    assert normal.card_reward_group_delta == 1
    assert normal.card_reward_group_count == 1
    assert normal.effects[0].kind == "reward_extra_card_group"
    assert normal.effects[0].metadata["card_choice_count"] == 3
    assert elite.card_reward_group_delta == 0
    assert elite.effects == ()
    assert event.card_reward_group_delta == 0
    assert event.effects == ()


def test_lava_rock_adds_extra_act_one_boss_relic_reward() -> None:
    act_one_boss = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="boss",
            relic_count=1,
            owned_relics=("lava_rock",),
            metadata={"act": 1},
        )
    )
    act_two_boss = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="boss",
            relic_count=1,
            owned_relics=("lava_rock",),
            metadata={"act": 2},
        )
    )
    elite = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="elite",
            relic_count=1,
            owned_relics=("lava_rock",),
            metadata={"act": 1},
        )
    )

    assert act_one_boss.relic_count_delta == 1
    assert act_one_boss.relic_count == 2
    assert [(effect.kind, effect.amount, effect.content_id) for effect in act_one_boss.effects] == [
        ("reward_extra_relic", 1, "lava_rock")
    ]
    assert act_one_boss.effects[0].metadata["metadata_equals"] == {"act": 1}
    assert act_two_boss.effects == ()
    assert elite.effects == ()


def test_gold_and_potion_reward_relics_apply_to_combat_rewards() -> None:
    result = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="combat",
            encounter_type="normal",
            card_choice_count=3,
            owned_relics=("amethyst_aubergine", "white_beast_statue"),
        )
    )
    event = resolve_reward_generation_triggers(
        RewardGenerationContext(
            source="event",
            encounter_type="normal",
            card_choice_count=3,
            owned_relics=("amethyst_aubergine", "white_beast_statue"),
        )
    )

    assert result.gold_delta == 15
    assert result.potion_count_delta == 1
    assert result.potion_count == 1
    assert [(effect.kind, effect.amount, effect.content_id) for effect in result.effects] == [
        ("reward_gold_delta", 15, "amethyst_aubergine"),
        ("reward_extra_potion", 1, "white_beast_statue"),
    ]
    assert event.gold_delta == 0
    assert event.potion_count_delta == 0


def test_reward_modifier_specs_provide_generic_future_marker_plumbing() -> None:
    result = resolve_reward_generation_triggers(
        RewardGenerationContext(source="event", card_choice_count=0, potion_count=1),
        modifiers=(
            RewardModifierSpec(
                content_id="test_reward_modifier",
                source_kind="custom",
                kind="reward_extra_potion",
                potion_count_delta=1,
                metadata={"reason": "unit_test"},
            ),
        ),
    )

    assert result.potion_count_delta == 1
    assert result.potion_count == 2
    assert result.effects[0].source_kind == "custom"
    assert result.effects[0].content_id == "test_reward_modifier"
    assert result.effects[0].metadata["potion_count_delta"] == 1


def test_reward_generation_context_dispatches_custom_trigger_handlers() -> None:
    def handler(context: TriggerContext) -> TriggerHandlerResult:
        if context.metadata["reward_source"] != "combat":
            return TriggerHandlerResult()
        return TriggerHandlerResult(
            effects=(
                TriggerEffect(
                    kind="reward_marker_from_dispatch",
                    trigger=context.trigger,
                    source_kind="custom",
                    content_id="test_dispatcher",
                    target_id="reward",
                    metadata={"potion_count": context.metadata["potion_count"]},
                ),
            )
        )

    dispatcher = TriggerDispatcher(
        (
            TriggerHandler(
                handler_id="test_reward_dispatcher",
                trigger=GameTrigger.COMBAT_REWARD_GENERATED,
                callback=handler,
            ),
        )
    )

    result = resolve_reward_generation_triggers(
        RewardGenerationContext(source="combat", potion_count=2),
        dispatcher=dispatcher,
    )

    assert result.effects[-1].kind == "reward_marker_from_dispatch"
    assert result.effects[-1].metadata == {"potion_count": 2}
