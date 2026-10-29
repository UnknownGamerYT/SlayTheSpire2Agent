from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardInstance, CombatState, MonsterState, PlayerState, RunState
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import _apply_shuffle_all_cards_to_draw_effect, _draw_cards


def _card(instance_id: str) -> CardInstance:
    return CardInstance(
        instance_id=instance_id, card_id="defend", type="skill", target="self", effects={"block": 5}
    )


def _combat() -> CombatState:
    return CombatState(
        player=PlayerState(hp=80, max_hp=80), discard_pile=(_card("discard:1"),)
    )


def test_soot_is_inserted_during_shuffle_and_can_be_drawn_by_the_same_draw() -> None:
    result, _rng, events = _draw_cards(_combat(), new_rng_state(5), 2, relics=("BIIIG_HUG",))

    assert sorted(card.card_id for card in result.hand) == ["defend", "soot"]
    soot = next(card for card in result.hand if card.card_id == "soot")
    assert soot.type.value == "status"
    assert soot.cost == -1
    assert soot.custom["unplayable"] is True
    assert result.metadata["biiig_hug_soot_count"] == 1
    assert len([event for event in events if event.source_id == "biiig_hug"]) == 1


@pytest.mark.parametrize("kind", ["no_relic", "draw_without_shuffle", "empty_piles"])
def test_biiig_hug_does_not_create_soot_without_a_real_shuffle(kind: str) -> None:
    combat = _combat()
    relics = ("biiig_hug",)
    if kind == "no_relic":
        relics = ()
    elif kind == "draw_without_shuffle":
        combat = combat.model_copy(update={"draw_pile": (_card("draw:1"),)})
    else:
        combat = combat.model_copy(update={"discard_pile": ()})

    result, _rng, events = _draw_cards(combat, new_rng_state(7), 1, relics=relics)

    assert not any(card.card_id == "soot" for card in (*result.hand, *result.draw_pile))
    assert not any(event.source_id == "biiig_hug" for event in events)


def test_repeated_shuffles_keep_soot_instances_distinct_across_serialization() -> None:
    combat, rng, _events = _draw_cards(_combat(), new_rng_state(5), 2, relics=("biiig_hug",))
    combat = combat.model_copy(update={"discard_pile": combat.hand, "hand": ()})
    state = RunState(
        seed=5, character_id="TEST", rng=rng, player=combat.player,
        relics=("biiig_hug",), combat=combat,
    )
    restored = load_state(serialize(state))
    assert restored.combat is not None

    first = _draw_cards(combat, rng, 3, relics=state.relics)
    second = _draw_cards(restored.combat, restored.rng, 3, relics=restored.relics)

    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[1] == second[1]
    assert [event.model_dump(mode="json") for event in first[2]] == [
        event.model_dump(mode="json") for event in second[2]
    ]
    soot = [card for card in first[0].hand if card.card_id == "soot"]
    assert len(soot) == 2
    assert len({card.instance_id for card in soot}) == 2
    assert first[0].metadata["biiig_hug_soot_count"] == 2


def test_explicit_shuffle_all_cards_adds_one_soot_without_recursive_shuffling() -> None:
    source = _card("shuffle-card")
    combat = _combat().model_copy(update={"hand": (_card("hand:1"),)})

    result, _rng, events = _apply_shuffle_all_cards_to_draw_effect(
        combat, new_rng_state(8), source, {"shuffle_all_cards_to_draw": True},
        relics=("biiig_hug",),
    )

    assert len(result.draw_pile) == 3
    assert sum(card.card_id == "soot" for card in result.draw_pile) == 1
    assert result.hand == result.discard_pile == ()
    assert sum(event.source_id == "biiig_hug" for event in events) == 1


def test_playing_a_draw_card_replays_soot_and_other_shuffle_relics_deterministically() -> None:
    draw_card = CardInstance(
        instance_id="draw:2", card_id="draw_two", type="skill", cost=0,
        target="self", effects={"draw": 2},
    )
    combat = _combat().model_copy(update={
        "hand": (draw_card,),
        "monsters": (MonsterState(monster_id="dummy", hp=100, max_hp=100),),
    })
    state = RunState(
        seed=5, character_id="TEST", rng=new_rng_state(5), player=combat.player,
        relics=("biiig_hug", "the_abacus"), combat=combat,
    )
    action = next(action for action in legal_actions(state) if action.type == "play_card")

    result = step(state, action)
    replayed = step(load_state(serialize(state)), action)

    assert serialize(result) == serialize(replayed)
    assert result.combat is not None
    assert result.combat.player.block == 6  # Abacus still receives the same shuffle event.
    assert any(card.card_id == "soot" for card in result.combat.hand)
    assert sum(event.source_id == "biiig_hug" for event in result.replay_log[-1].events) == 1
