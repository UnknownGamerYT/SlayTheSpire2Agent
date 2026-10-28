from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _state as _fight
from combat.test_glam import _play

from sts2sim import load_state, serialize
from sts2sim.engine import Action, CardInstance, CombatState, EffectEvent, PlayerState, RunState
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import (
    _apply_deck_add_relics,
    _enchant_card,
    _upgrade_card_instance,
)


def _state(*relics: str) -> RunState:
    return RunState(
        seed=41, character_id="TEST", rng=new_rng_state(41),
        player=PlayerState(hp=30, max_hp=80, gold=10), relics=tuple(relics),
    )


def _card(instance_id: str, *, block: bool = False) -> CardInstance:
    return CardInstance(
        instance_id=instance_id, card_id="defend" if block else "strike",
        type="skill" if block else "attack", target="self" if block else "enemy",
        effects={"block": 5} if block else {"damage": 6},
    )


def _add(
    state: RunState, cards: tuple[CardInstance, ...], *, kind: str = "reward_card_taken"
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    after = state.model_copy(update={"master_deck": state.master_deck + cards})
    events = tuple(EffectEvent(kind=kind, target_id=card.instance_id) for card in cards)
    return _apply_deck_add_relics(
        state, after, Action(type="take_reward_card", target_id="reward:card:0"), events
    )


def test_bing_bong_duplicates_each_original_once_and_preserves_card_properties() -> None:
    state = _state("BING_BONG")
    first = _card("first", block=True).model_copy(update={"upgraded": True, "custom": {"x": 1}})
    first = _enchant_card(first, "Nimble", 1, source_id="event")
    second = _card("second")

    result, events = _add(state, (first, second))

    assert len(result.master_deck) == 4
    assert len({card.instance_id for card in result.master_deck}) == 4
    for original, duplicate in zip(result.master_deck[:2], result.master_deck[2:], strict=True):
        assert original.model_dump(exclude={"instance_id"}) == duplicate.model_dump(
            exclude={"instance_id"}
        )
    assert sum(event.source_id == "bing_bong" for event in events) == 2


def test_bing_bong_copies_count_for_gold_healing_and_each_gets_one_fresnel_enchant() -> None:
    state = _state("bing_bong", "lucky_fysh", "book_of_five_rings", "fresnel_lens")
    state = state.model_copy(update={"flags": {"relic_counters": {"book_of_five_rings": 3}}})

    result, events = _add(state, (_card("guard", block=True),))

    assert result.player.gold == 40  # 15 for each card, including the Bing Bong copy.
    assert result.player.hp == 50
    assert result.flags["relic_counters"]["book_of_five_rings"] == 0
    assert [card.effects["block"] for card in result.master_deck] == [5, 5]
    combat = _fight(*result.master_deck)
    for card in result.master_deck:
        combat = _play(combat, card.instance_id)
    assert combat.player.block == 14  # Each native5 gains its own Nimble2 once.
    assert [len(card.enchantments) for card in result.master_deck] == [1, 1]
    assert [event.amount for event in events if event.kind == "gold_gained"] == [15, 15]


def test_book_of_five_rings_tracks_multiple_cycles_across_save_and_load() -> None:
    state = _state("book_of_five_rings")
    state, _events = _add(state, tuple(_card(f"card:{index}") for index in range(4)))
    assert state.player.hp == 30
    assert state.flags["relic_counters"]["book_of_five_rings"] == 4
    state = load_state(serialize(state))

    state, events = _add(state, tuple(_card(f"new:{index}") for index in range(7)))

    assert state.player.hp == 70
    assert state.flags["relic_counters"]["book_of_five_rings"] == 1
    assert [event.amount for event in events if event.kind == "player_healed"] == [20, 20]


@pytest.mark.parametrize(
    "kind", ["event_card_transformed", "card_upgraded", "card_created", "card_removed"]
)
def test_transforms_upgrades_and_combat_creations_are_not_deck_additions(kind: str) -> None:
    state = _state("bing_bong", "lucky_fysh", "book_of_five_rings", "fresnel_lens")
    after = state.model_copy(update={"master_deck": (_card("replacement", block=True),)})
    events = (EffectEvent(kind=kind, target_id="replacement"),)

    result, result_events = _apply_deck_add_relics(
        state, after, Action(type="choose_event", target_id="transform"), events
    )

    assert result == after
    assert result_events == events


@pytest.mark.parametrize(
    "kind", ["ancient_card_added", "event_custom_card_created", "event_card_added",
             "event_random_card_added", "relic_deck_card_added", "relic_deck_card_duplicated"]
)
def test_all_run_addition_sources_trigger_lucky_fysh(kind: str) -> None:
    state, events = _add(_state("lucky_fysh"), (_card("new-card"),), kind=kind)
    assert state.player.gold == 25
    assert events[-1].kind == "gold_gained"


def test_shop_card_addition_uses_purchase_card_instance_metadata() -> None:
    state = _state("lucky_fysh")
    after = state.model_copy(update={"master_deck": (_card("shop:1"),)})
    purchase = EffectEvent(
        kind="shop_item_bought", target_id="strike", amount=-50,
        metadata={"card_instance_id": "shop:1", "item_kind": "card"},
    )

    result, _events = _apply_deck_add_relics(
        state, after, Action(type="shop_buy", target_id="shop:0"), (purchase,)
    )

    assert result.player.gold == 25


def test_addition_with_reused_id_is_repaired_and_copy_ids_stay_unique_after_removal() -> None:
    state = _state("bing_bong")
    old = _card("card_002")
    state = state.model_copy(update={"master_deck": (old,)})
    state, events = _add(state, (_card("card_002", block=True),))
    assert len({card.instance_id for card in state.master_deck}) == 3
    new_id = state.master_deck[1].instance_id
    assert events[0].target_id == new_id
    copy_id = state.master_deck[2].instance_id
    state = load_state(serialize(state.model_copy(update={"master_deck": (old,)})))

    state, _events = _add(state, (_card("next"),))

    assert state.master_deck[-1].instance_id != copy_id


def test_combat_addition_keeps_current_counters_and_player_state_in_sync() -> None:
    state = _state("book_of_five_rings", "lucky_fysh")
    state = state.model_copy(update={
        "flags": {"relic_counters": {"book_of_five_rings": 4, "nunchaku": 1}},
        "combat": CombatState(player=state.player, metadata={
            "relic_counters": {"book_of_five_rings": 4, "nunchaku": 7},
        }),
    })

    result, _events = _add(state, (_card("new"),), kind="relic_deck_card_added")

    assert result.combat is not None
    assert result.combat.player == result.player
    assert result.player.hp == 50
    assert result.flags["relic_counters"]["nunchaku"] == 7
    assert result.combat.metadata["relic_counters"]["book_of_five_rings"] == 0


def test_multiple_additions_with_the_same_legacy_id_get_separate_event_targets() -> None:
    first = _card("same-id")
    second = _card("same-id", block=True)

    result, events = _add(_state("bing_bong"), (first, second))

    assert len(result.master_deck) == 4
    assert len({card.instance_id for card in result.master_deck}) == 4
    assert events[0].target_id == result.master_deck[0].instance_id
    assert events[1].target_id == result.master_deck[1].instance_id


def test_fresnel_enchants_only_block_cards_including_nested_and_formula_effects() -> None:
    state = _state("fresnel_lens")
    nested = _card("nested").model_copy(update={"effects": {"sequence": [{"block": 3}]}})
    formula = _card("formula").model_copy(update={
        "effects": {"block_formula": {"formula": "discard_pile_count"}}
    })

    result, _events = _add(state, (_card("attack"), nested, formula))

    assert result.master_deck[0].enchantments == ()
    assert result.master_deck[1].effects["sequence"][0]["block"] == 3
    assert "bonus" not in result.master_deck[2].effects["block_formula"]
    combat = _play(_fight(*result.master_deck), "nested")
    assert combat.player.block == 5  # Native3 + Nimble2.
    combat = _play(combat, "formula")
    assert combat.player.block == 8  # One discarded card + Nimble2.


def test_upgrading_fresnel_card_preserves_its_actual_nimble_block_bonus() -> None:
    card = _card("guard", block=True).model_copy(update={"custom": {"source_spec": {
        "id": "defend", "type": "skill", "target": "self", "block": 5,
        "description": "Gain 5 [gold]Block[/gold].", "vars": {"Block": 5},
        "upgrade": {"block": "+3"},
    }}})
    card = _enchant_card(card, "Nimble", 2, source_id="fresnel_lens")

    upgraded = _upgrade_card_instance(card)

    assert upgraded.upgraded is True
    assert upgraded.effects["sequence"][0]["block"] == 8
    assert len(upgraded.enchantments) == 1
    assert _play(_fight(upgraded), upgraded.instance_id).player.block == 10
