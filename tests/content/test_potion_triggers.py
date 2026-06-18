from __future__ import annotations

from sts2sim.mechanics.potion_triggers import (
    PotionUseModifierSpec,
    PotionUseTriggerContext,
    resolve_potion_use_triggers,
)
from sts2sim.mechanics.triggers import (
    GameTrigger,
    TriggerContext,
    TriggerDispatcher,
    TriggerEffect,
    TriggerHandler,
    TriggerHandlerResult,
)


def test_toy_ornithopter_emits_capped_heal_marker_on_potion_use() -> None:
    result = resolve_potion_use_triggers(
        PotionUseTriggerContext(
            potion={"id": "FIRE_POTION"},
            slot_index=0,
            target_id="jaw_worm",
            player_hp=47,
            player_max_hp=50,
            owned_relics=("TOY_ORNITHOPTER",),
        )
    )

    assert result.hp_delta == 3
    assert result.effects[0].kind == "potion_use_heal"
    assert result.effects[0].amount == 3
    assert result.effects[0].content_id == "toy_ornithopter"
    assert result.effects[0].metadata["potion_id"] == "fire_potion"
    assert result.effects[0].metadata["potion_slot"] == "potion:0"
    assert result.trigger_resolution.trigger is GameTrigger.POTION_USED


def test_potion_use_context_dispatches_custom_trigger_handlers() -> None:
    def handler(context: TriggerContext) -> TriggerHandlerResult:
        if context.metadata["potion_id"] != "swift_potion":
            return TriggerHandlerResult()
        return TriggerHandlerResult(
            energy_delta=1,
            effects=(
                TriggerEffect(
                    kind="potion_used_custom_energy",
                    trigger=context.trigger,
                    source_kind="custom",
                    content_id="test_potion_handler",
                    amount=1,
                    target_id="player",
                    metadata={"slot": context.metadata["potion_slot"]},
                ),
            ),
        )

    dispatcher = TriggerDispatcher(
        (
            TriggerHandler(
                handler_id="test_potion_dispatcher",
                trigger=GameTrigger.POTION_USED,
                callback=handler,
            ),
        )
    )

    result = resolve_potion_use_triggers(
        PotionUseTriggerContext(potion="SWIFT_POTION", slot_id="potion:1"),
        dispatcher=dispatcher,
    )

    assert result.energy_delta == 1
    assert result.effects[0].kind == "potion_used_custom_energy"
    assert result.effects[0].metadata == {"slot": "potion:1"}


def test_generic_potion_modifier_specs_emit_markers_and_deltas() -> None:
    result = resolve_potion_use_triggers(
        PotionUseTriggerContext(potion="block_potion"),
        modifiers=(
            PotionUseModifierSpec(
                content_id="test_potion_modifier",
                source_kind="power",
                kind="potion_use_block_bonus",
                block_delta=4,
                metadata={"reason": "unit_test"},
            ),
        ),
    )

    assert result.block_delta == 4
    assert result.effects[0].source_kind == "power"
    assert result.effects[0].content_id == "test_potion_modifier"
    assert result.effects[0].amount == 4
    assert result.effects[0].metadata["block_delta"] == 4
