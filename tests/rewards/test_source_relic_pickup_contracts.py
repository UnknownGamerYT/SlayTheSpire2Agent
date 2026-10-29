"""Direct-source pickup behavior and isolated model/player RNG families."""

from __future__ import annotations

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    ActionType,
    CardInstance,
    CombatState,
    RunPhase,
    load_state,
    serialize_state,
    transitions,
)
from sts2sim.engine.rng import capture_random_state, random_from_state
from sts2sim.mechanics.relics import RelicEffectMarker


@pytest.mark.parametrize("character", ["IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT"])
def test_arcane_scroll_adds_one_rare_directly_without_upgrade_roll_or_choice(character) -> None:
    state = new_run("ARCANE-SCROLL", character, 0)
    state = state.model_copy(update={"phase": RunPhase.MAP})
    before = state.rng.streams.get("rewards")
    count = len(state.master_deck)
    actual, _ = transitions._apply_relic_pickup_effects(state, "arcane_scroll")
    assert len(actual.master_deck) == count + 1 and actual.reward is None
    selected = transitions._reward_card_spec(actual, actual.master_deck[-1].card_id)
    assert selected["rarity"].lower() == "rare"
    assert actual.rng.streams["rewards"].draws - (before.draws if before else 0) == 1


def test_arcane_scroll_still_allows_egg_upgrade_hooks_without_random_upgrade_draw() -> None:
    state = new_run("ARCANE-EGG", "IRONCLAD", 0, source_data={
        "relics": ["molten_egg"], "combat_reward_card_pool": [
            {"id": "FEED", "color": "ironclad", "rarity": "Rare", "type": "Attack"},
        ],
    })
    actual, _ = transitions._apply_relic_pickup_effects(state, "arcane_scroll")
    assert actual.master_deck[-1].card_id == "feed" and actual.master_deck[-1].upgraded


def test_byrdpip_transforms_every_deck_and_combat_egg_and_does_not_add_extra_swoop() -> None:
    state = new_run("BYRDPIP", "REGENT", 0)
    eggs = tuple(CardInstance(instance_id=f"egg_{i}", card_id="byrdonis_egg") for i in range(3))
    ordinary = CardInstance(instance_id="ordinary", card_id="strike_regent")
    combat = CombatState(player=state.player, hand=(eggs[0],), draw_pile=(eggs[1],),
                         discard_pile=(eggs[2],))
    state = state.model_copy(update={"phase": RunPhase.COMBAT, "master_deck": (*eggs, ordinary),
                                     "combat": combat})
    actual, _ = transitions._apply_relic_pickup_effects(state, "byrdpip")
    assert len(actual.master_deck) == 4
    assert [c.card_id for c in actual.master_deck] == ["byrd_swoop"] * 3 + ["strike_regent"]
    assert actual.combat is not None
    assert all(c.card_id == "byrd_swoop" for zone in ("hand", "draw_pile", "discard_pile")
               for c in getattr(actual.combat, zone))
    assert tuple(c.instance_id for c in actual.master_deck[:3]) == tuple(
        c.instance_id for c in eggs
    )


def test_astrolabe_chosen_transform_uses_niche_and_preserves_public_reload() -> None:
    state = new_run("ASTROLABE", "IRONCLAD", 0, source_data={
        "deck": [{"card_id": "original", "cost": 1}],
        "cards": [{"id": "first", "color": "ironclad", "rarity": "Common"},
                  {"id": "second", "color": "ironclad", "rarity": "Common"}],
    })
    state, _ = transitions._apply_relic_pickup_effects(state, "astrolabe")
    action = next(a for a in legal_actions(state) if a.type == ActionType.CHOOSE_CARD)
    before = state.rng.streams.get("niche")
    actual = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(actual) == serialize_state(restored)
    assert actual.master_deck[0].card_id in {"first", "second"}
    assert actual.master_deck[0].upgraded
    assert actual.rng.streams["niche"].draws - (before.draws if before else 0) == 1


@pytest.mark.parametrize("tungsten,expected_loss", [(False, 15), (True, 14)])
def test_fragrant_mushroom_damage_is_unblockable_and_uses_damage_relic_modifiers(
    tungsten, expected_loss,
) -> None:
    state = new_run("FRAGRANT", "IRONCLAD", 0, source_data={
        "relics": ["tungsten_rod"] if tungsten else [],
        "deck": [{"card_id": card_id} for card_id in ("anger", "armaments", "bash", "body_slam")],
    })
    state = state.model_copy(update={"player": state.player.model_copy(update={"block": 20})})
    actual, _ = transitions._apply_relic_pickup_effects(state, "fragrant_mushroom")
    assert state.player.hp - actual.player.hp == expected_loss
    assert actual.player.block == 20
    assert sum(card.upgraded for card in actual.master_deck) == 2


def test_fragrant_mushroom_stable_shuffle_is_permutation_invariant_and_consumes_niche() -> None:
    state = new_run("FRAGRANT-SHUFFLE", "IRONCLAD", 0, source_data={
        "deck": [{"card_id": card_id} for card_id in ("anger", "armaments", "bash", "body_slam")],
    })
    marker = RelicEffectMarker(kind="upgrade_deck_cards", relic_id="fragrant_mushroom", amount=2,
                               metadata={"selection": "random"})
    selected = []
    for deck in (state.master_deck, tuple(reversed(state.master_deck))):
        rng = random_from_state(state.rng)
        actual, _ = transitions._apply_relic_upgrade_deck_cards_marker(
            state.model_copy(update={"master_deck": deck}), rng, "fragrant_mushroom", marker,
        )
        selected.append({card.card_id for card in actual.master_deck if card.upgraded})
        assert capture_random_state(rng).streams["niche"].draws == 3
    assert selected[0] == selected[1]
