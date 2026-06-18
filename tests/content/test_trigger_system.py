from __future__ import annotations

from sts2sim.mechanics import (
    GameTrigger,
    TriggerContext,
    TriggerDispatcher,
    TriggerEffect,
    TriggerHandler,
    TriggerHandlerResult,
    game_trigger,
    resolve_game_trigger,
)


def test_game_trigger_aliases_normalize_to_shared_names() -> None:
    assert game_trigger("start_combat") is GameTrigger.COMBAT_START
    assert game_trigger("start-of-combat") is GameTrigger.COMBAT_START
    assert game_trigger("shop_enter") is GameTrigger.SHOP_ENTERED


def test_combat_relics_are_available_through_unified_trigger() -> None:
    result = resolve_game_trigger(
        GameTrigger.COMBAT_START,
        relics=("ANCHOR", "VAJRA"),
        player_hp=70,
        player_max_hp=80,
        encounter_type="normal",
    )

    assert result.trigger is GameTrigger.COMBAT_START
    assert result.combat_relic_resolution is not None
    assert result.block_delta == 10
    assert [(effect.kind, effect.amount, effect.content_id) for effect in result.effects] == [
        ("gain_block", 10, "anchor"),
        ("gain_status", 1, "vajra"),
    ]
    assert result.effects[1].metadata == {"hook": "start_of_combat", "status": "strength"}


def test_room_relics_are_available_through_unified_trigger() -> None:
    result = resolve_game_trigger(
        "shop_entered",
        relics=("MEAL_TICKET",),
        player_hp=73,
        player_max_hp=80,
    )

    assert result.trigger is GameTrigger.SHOP_ENTERED
    assert result.relic_hook_resolution is not None
    assert result.hp_delta == 7
    assert [(effect.kind, effect.amount, effect.content_id) for effect in result.effects] == [
        ("meal_ticket_healed", 7, "meal_ticket"),
    ]


def test_trigger_blockers_adapt_from_inferred_combat_relic_hooks() -> None:
    relic = {
        "id": "ODD_CLOCK",
        "name": "Odd Clock",
        "description": "At the start of each combat, do a precise bespoke thing.",
    }

    result = resolve_game_trigger(GameTrigger.COMBAT_START, relics=(relic,))

    assert len(result.blockers) == 1
    assert result.blockers[0].trigger is GameTrigger.COMBAT_START
    assert result.blockers[0].content_id == "odd_clock"
    assert "No pure combat relic helper" in result.blockers[0].reason


def test_custom_handlers_can_register_against_the_same_trigger_vocabulary() -> None:
    def free_attack_handler(context: TriggerContext) -> TriggerHandlerResult:
        if context.card_type != "attack":
            return TriggerHandlerResult()
        return TriggerHandlerResult(
            energy_delta=1,
            effects=(
                TriggerEffect(
                    kind="gain_energy",
                    trigger=context.trigger,
                    source_kind="power",
                    content_id="test_attack_refund",
                    amount=1,
                    target_id="player",
                    metadata={"card_id": context.card_id},
                ),
            ),
        )

    dispatcher = TriggerDispatcher(
        (
            TriggerHandler(
                handler_id="test_attack_refund",
                trigger=GameTrigger.CARD_PLAYED,
                source_kind="power",
                content_id="test_attack_refund",
                callback=free_attack_handler,
            ),
        )
    )

    result = resolve_game_trigger(
        GameTrigger.CARD_PLAYED,
        context=TriggerContext(
            GameTrigger.CARD_PLAYED,
            card_type="attack",
            card_id="strike",
        ),
        dispatcher=dispatcher,
    )

    assert result.energy_delta == 1
    assert result.effects[0].source_kind == "power"
    assert result.effects[0].metadata == {"card_id": "strike"}
