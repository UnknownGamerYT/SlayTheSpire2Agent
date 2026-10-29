"""v0.107.1 entry source: all 35 overrides and runtime fact adapters."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    CardEnchantment,
    CardInstance,
    CardType,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
)
from sts2sim.engine.transitions import _event_eligibility_context, _source_total_floor
from sts2sim.mechanics.event_eligibility import EventEligibilityContext, evaluate_event_eligibility
from sts2sim.mechanics.event_entry_data import SOURCE_EVENT_IDS

SOURCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/stable_event_entry_source.json").read_text()
)


def context(**changes: Any) -> EventEligibilityContext:
    return replace(
        EventEligibilityContext(
            act=2,
            floor=10,
            total_floor=20,
            gold=200,
            hp=50,
            max_hp=80,
            removable_cards=3,
            removable_basic_cards=2,
            strikes=2,
            defends=2,
            potions=("FIRE_POTION", "BLOCK_POTION"),
            player_count=1,
            has_event_pet=False,
            unlocked_character_count=5,
            tradeable_relic_count=5,
            deck_size=10,
            transformable_cards=3,
            perfect_fit_cards=3,
            souls_power_cards=3,
            spiral_cards=3,
            has_available_relics=True,
        ),
        **changes,
    )


CASES: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ("AMALGAMATOR", {}, {"defends": 1}),
    ("BRAIN_LEECH", {"act": 2}, {"act": 3}),
    ("BYRDONIS_NEST", {}, {"has_event_pet": True}),
    ("COLORFUL_PHILOSOPHERS", {"unlocked_character_count": 2}, {"unlocked_character_count": 1}),
    ("COLOSSAL_FLOWER", {"hp": 19}, {"hp": 18}),
    ("CRYSTAL_SPHERE", {"gold": 100}, {"gold": 99}),
    ("DENSE_VEGETATION", {"hp": 1}, {"player_count": 2, "hp": 8}),
    ("DOLL_ROOM", {}, {"act": 1}),
    ("ENDLESS_CONVEYOR", {"gold": 120}, {"gold": 119}),
    ("FAKE_MERCHANT", {"gold": 0, "potions": ("FOUL_POTION",)}, {"gold": 99}),
    ("FIELD_OF_MAN_SIZED_HOLES", {"perfect_fit_cards": 1}, {"perfect_fit_cards": 0}),
    ("GRAVE_OF_THE_FORGOTTEN", {"souls_power_cards": 1}, {"souls_power_cards": 0}),
    ("LUMINOUS_CHOIR", {"gold": 149}, {"gold": 148}),
    ("MORPHIC_GROVE", {"transformable_cards": 2}, {"transformable_cards": 1}),
    ("POTION_COURIER", {}, {"act": 1}),
    ("PUNCH_OFF", {"total_floor": 6}, {"total_floor": 5}),
    ("RANWID_THE_ELDER", {"tradeable_relic_count": 1}, {"tradeable_relic_count": 0}),
    ("RELIC_TRADER", {"tradeable_relic_count": 5}, {"tradeable_relic_count": 4}),
    ("ROOM_FULL_OF_CHEESE", {}, {"act": 3}),
    ("ROUND_TEA_PARTY", {"hp": 12}, {"hp": 11}),
    ("SLIPPERY_BRIDGE", {"total_floor": 7}, {"total_floor": 6}),
    ("SPIRALING_WHIRLPOOL", {"spiral_cards": 1}, {"spiral_cards": 0}),
    ("STONE_OF_ALL_TIME", {}, {"potions": ()}),
    ("SYMBIOTE", {}, {"act": 1}),
    ("TEA_MASTER", {"gold": 150}, {"gold": 149}),
    ("THE_FUTURE_OF_POTIONS", {}, {"potions": ("FIRE_POTION",)}),
    ("THE_LEGENDS_WERE_TRUE", {"act": 1}, {"act": 1, "deck_size": 0}),
    ("TRASH_HEAP", {"hp": 6}, {"hp": 5}),
    ("UNREST_SITE", {"hp": 56}, {"hp": 57}),
    ("WATERLOGGED_SCRIPTORIUM", {"gold": 55}, {"gold": 54}),
    ("WELCOME_TO_WONGOS", {"gold": 100}, {"gold": 99}),
    ("WHISPERING_HOLLOW", {"gold": 44}, {"gold": 43}),
    ("WOOD_CARVINGS", {"removable_basic_cards": 1}, {"removable_basic_cards": 0}),
    ("ZEN_WEAVER", {"gold": 125}, {"gold": 124}),
)


def test_source_projection_covers_every_override_and_all_canonical_events() -> None:
    assert frozenset(SOURCE["canonical_event_ids"]) == SOURCE_EVENT_IDS
    assert len(SOURCE_EVENT_IDS) == 57
    assert {value[0] for value in CASES} | {"WAR_HISTORIAN_REPY"} == set(
        SOURCE["source_entry_predicates"]
    )


@pytest.mark.parametrize("event_id,allowed,blocked", CASES)
def test_every_source_entry_override_boundaries(
    event_id: str,
    allowed: dict[str, Any],
    blocked: dict[str, Any],
) -> None:
    assert evaluate_event_eligibility({"id": event_id}, context(**allowed)).eligible
    assert not evaluate_event_eligibility({"id": event_id}, context(**blocked)).eligible


def test_inherited_true_and_disabled_event_are_independent_of_stale_api_clauses() -> None:
    for event_id in SOURCE_EVENT_IDS - set(SOURCE["source_entry_predicates"]):
        assert evaluate_event_eligibility({"id": event_id}, context()).eligible
    assert not evaluate_event_eligibility({"id": "WAR_HISTORIAN_REPY"}, context()).eligible
    assert evaluate_event_eligibility(
        {
            "id": "LUMINOUS_CHOIR",
            "preconditions": ["Requires 100-149 gold"],
        },
        context(gold=1000),
    ).eligible
    assert not evaluate_event_eligibility(
        {"id": "LUMINOUS_CHOIR"},
        context(
            gold=1000,
            has_available_relics=False,
        ),
    ).eligible


def test_all_player_checks_apply_supplied_facts_and_missing_facts_stay_unknown() -> None:
    peer = context(gold=99, player_count=2)
    local = context(player_count=2, other_players=(peer,))
    assert not evaluate_event_eligibility({"id": "CRYSTAL_SPHERE"}, local).eligible
    peer = replace(peer, gold=100)
    assert evaluate_event_eligibility(
        {"id": "CRYSTAL_SPHERE"},
        replace(
            local,
            other_players=(peer,),
        ),
    ).eligible
    unknown = evaluate_event_eligibility({"id": "CRYSTAL_SPHERE"}, replace(local, other_players=()))
    assert unknown.eligible and unknown.unknown
    peer = replace(peer, gold=0, unavailable_player_facts=frozenset({"gold"}))
    unknown = evaluate_event_eligibility(
        {"id": "CRYSTAL_SPHERE"},
        replace(
            local,
            other_players=(peer,),
        ),
    )
    assert unknown.eligible and unknown.unknown


def card(
    card_id: str,
    *,
    basic: bool = True,
    exhausts: bool = False,
    tags: tuple[str, ...] = (),
    enchanted: bool = False,
) -> CardInstance:
    return CardInstance(
        instance_id=card_id,
        card_id=card_id,
        type=CardType.ATTACK,
        tags=tags,
        exhausts=exhausts,
        enchantments=(CardEnchantment(keyword="sharp", amount=1),) if enchanted else (),
        custom={"source_spec": {"rarity": "Basic" if basic else "Common"}},
    )


def test_runtime_adapter_uses_basic_tags_removability_and_actual_enchantment_rules() -> None:
    state = new_run("CARDS", "IRONCLAD", 0)
    state = state.model_copy(
        update={
            "master_deck": (
                card("renamed_strike", tags=("Strike",)),
                card("eternal_basic", tags=("Strike", "Eternal")),
                card("common_pommel", basic=False, tags=("Strike",)),
                card("exhausting_skill", exhausts=True),
                card("already_enchanted", exhausts=True, enchanted=True),
                card("unplayable", tags=("Unplayable",)),
            )
        }
    )
    facts = _event_eligibility_context(state)
    assert facts.strikes == 1
    assert facts.transformable_cards == 5
    assert facts.perfect_fit_cards == 4
    assert facts.souls_power_cards == 1
    assert facts.spiral_cards == 2  # Eternal still permits enchantments.


def test_runtime_adapter_derives_pets_tradeability_and_relic_availability() -> None:
    state = new_run("FACTS", "IRONCLAD", 0)
    assert _event_eligibility_context(state).tradeable_relic_count == 0  # Starter relic.
    state = state.model_copy(
        update={
            "relics": (
                "anchor",
                "old_coin",
                "lizard_tail",
                "maw_bank",
                "byrdpip",
            ),
            "flags": {
                **state.flags,
                "melted_relic_ids": ("anchor",),
                "relic_counters": {"lizard_tail": 0},
                "maw_bank_disabled": True,
                "player_relic_bag": {"Rare": []},
            },
        }
    )
    facts = _event_eligibility_context(state)
    assert facts.has_event_pet
    assert facts.tradeable_relic_count == 0
    assert facts.has_available_relics is False
    state = state.model_copy(update={"relics": ("custom_unknown_relic",)})
    assert _event_eligibility_context(state).tradeable_relic_count is None
    state = state.model_copy(update={"relics": (), "master_deck": (card("BYRDONIS_EGG"),)})
    assert _event_eligibility_context(state).has_event_pet


def enter_event(state: Any, event_id: str, *, coordinate_floor: int = 2) -> Any:
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=3, kind=RoomKind.START)
    target = MapNodeState(
        node_id="event", act=state.act, floor=coordinate_floor, lane=3, kind=RoomKind.EVENT
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "ancient": None,
            "flags": {
                **state.flags,
                "event_pool": (event_id.lower(),),
                "resolve_unknown_rooms": False,
            },
            "map": MapState(
                act=state.act,
                nodes=(start, target),
                edges=(MapEdgeState(from_id="start", to_id="event"),),
                current_node_id="start",
                completed_node_ids=("start",),
            ),
        }
    )
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


def test_public_total_floor_counts_prior_rooms_and_ancient_instead_of_local_coordinate() -> None:
    base = new_run("FLOOR", "IRONCLAD", 0, source_data={"act_id": "UNDERDOCKS"})
    assert _source_total_floor(base) == 1
    early = enter_event(base, "PUNCH_OFF", coordinate_floor=12)
    assert early.event is not None and early.event.event_id == "UNKNOWN_EVENT"
    late = enter_event(
        base.model_copy(update={"room_history": tuple(f"room{i}" for i in range(5))}),
        "SLIPPERY_BRIDGE",
        coordinate_floor=2,
    )
    assert late.event is not None and late.event.event_id == "SLIPPERY_BRIDGE"
    assert _source_total_floor(late) == 7
    assert serialize_state(load_state(serialize_state(late))) == serialize_state(late)


def test_public_source_choir_accepts_rich_player_and_rejects_under_entry_price() -> None:
    base = new_run("CHOIR", "IRONCLAD", 0, source_data={"act_id": "OVERGROWTH"})
    for gold, expected in (
        (148, "UNKNOWN_EVENT"),
        (149, "LUMINOUS_CHOIR"),
        (1000, "LUMINOUS_CHOIR"),
    ):
        actual = enter_event(
            base.model_copy(
                update={
                    "player": base.player.model_copy(
                        update={"gold": gold},
                    )
                }
            ),
            "LUMINOUS_CHOIR",
        )
        assert actual.event is not None and actual.event.event_id == expected
