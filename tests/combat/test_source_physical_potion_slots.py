from __future__ import annotations

import json

import pytest
from test_named_monster_scripts import _combat

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import (
    CardInstance,
    EventOptionState,
    EventState,
    RewardState,
    RunPhase,
    ShopItemState,
    ShopState,
)
from sts2sim.engine.potion_inventory import inventory_slots, procure_at_slot
from sts2sim.engine.rng import capture_random_state, run_random_from_seed


def _choose(state, kind, slot=None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == kind
        and (slot is None or action.payload.get("potion_slot", action.target_id) == slot)
    )


def _source():
    state = new_run("PHYSICAL-POTIONS", "IRONCLAD", 0)
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "potions": ("blood_potion", "fruit_juice", "block_potion"),
            "flags": {**state.flags, "allow_free_potion_discard": True},
        }
    )


def _reload(state):
    restored = load_state(json.loads(json.dumps(serialize(state))))
    assert serialize(restored) == serialize(state)
    assert legal_actions(restored) == legal_actions(state)
    return restored


def test_use_discard_reload_and_reward_keep_other_physical_slots():
    state = _source()
    state = step(state, _choose(state, "use_potion", "potion:0"))
    assert inventory_slots(state) == (None, "fruit_juice", "block_potion")
    assert state.flags["potion_slot_indices"] == (1, 2)
    state = step(_reload(state), _choose(state, "discard_potion", "potion:1"))
    assert inventory_slots(state) == (None, None, "block_potion")
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(
                reward_id="physical", source="combat", potion_id="strength_potion"
            ),
        }
    )
    state = step(state, _choose(state, "take_reward_potion"))
    assert inventory_slots(state) == ("strength_potion", None, "block_potion")
    assert state.potions == ("strength_potion", "block_potion")
    assert state.flags["potion_slot_indices"] == (0, 2)
    _reload(state)


def test_preferred_slot_does_not_fill_earlier_holes_or_overwrite_and_sozu_precedes_index():
    state = _source().model_copy(
        update={
            "potions": ("block_potion",),
            "flags": {**_source().flags, "potion_slot_indices": (2,), "potion_slots": 7},
        }
    )
    before = serialize(state)
    state, events = procure_at_slot(state, "fire_potion", preferred_slot=5)
    assert inventory_slots(state) == (None, None, "block_potion", None, None, "fire_potion", None)
    assert events[0].metadata["potion_slot"] == "potion:5"
    blocked, events = procure_at_slot(state, "skill_potion", preferred_slot=5)
    assert serialize(blocked) == serialize(state)
    assert events[0].kind == "event_potion_skipped_no_slot"
    assert before != serialize(state)
    blocked = state.model_copy(update={"relics": ("sozu",)})
    same, events = procure_at_slot(blocked, "fire_potion", preferred_slot=99)
    assert same == blocked
    assert events[0].kind == "potion_procurement_blocked"


def test_entropic_brew_fills_first_holes_without_moving_a_held_potion():
    state = _source().model_copy(
        update={
            "potions": ("entropic_brew", "block_potion"),
            "flags": {**_source().flags, "potion_slot_indices": (0, 2)},
        }
    )
    state = step(state, _choose(state, "use_potion", "potion:0"))
    assert len(state.potions) == 3
    assert inventory_slots(state)[2] == "block_potion"
    generated = [e for e in state.replay_log[-1].events if e.kind == "potion_generated"]
    assert [e.metadata["potion_slot"] for e in generated] == ["potion:0", "potion:1"]
    _reload(state)


def _fight(potions, indices):
    state = _combat("TEST", moves=({"id": "ATTACK", "damage": {"normal": 999}},))
    return state.model_copy(
        update={
            "potions": potions,
            "rng": capture_random_state(run_random_from_seed("PHYSICAL-COMBAT")),
            "flags": {**state.flags, "potion_slot_indices": indices},
        }
    )


def test_fairy_consumption_between_damage_steps_does_not_shift_a_later_slot():
    state = _fight(("fairy_in_a_bottle", "fire_potion"), (1, 2))
    state = step(state, _choose(state, "end_turn"))
    assert state.player.hp > 0
    assert inventory_slots(state) == (None, None, "fire_potion")
    triggered = next(e for e in state.replay_log[-1].events if e.kind == "potion_passive_triggered")
    assert triggered.metadata["potion_slot"] == "potion:1"
    assert any(a.payload.get("potion_slot") == "potion:2" for a in legal_actions(state))
    _reload(state)


def test_alchemize_and_replayed_generation_keep_held_slot_identity():
    state = _fight(("fire_potion",), (2,))
    card = CardInstance(
        instance_id="generate",
        card_id="alchemize",
        type="skill",
        cost=0,
        target="self",
        effects={"add_random_potion": 1},
    )
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (card,)})})
    state = step(state, _choose(state, "play_card"))
    assert inventory_slots(state)[2] == "fire_potion"
    assert inventory_slots(state)[0] is not None
    assert state.flags["potion_slot_indices"] == (0, 2)
    generated = next(e for e in state.replay_log[-1].events if e.kind == "potion_generated")
    assert generated.metadata["potion_slot"] == "potion:0"
    _reload(state)


def test_invalid_explicit_inventory_fails_closed_and_old_saves_remain_contiguous():
    state = _source()
    assert inventory_slots(state) == state.potions
    for bad in ((0, 0, 2), (0, -1, 2), (0, 1), (0, 1, True)):
        with pytest.raises(ValueError, match="Potion slot indices"):
            inventory_slots(
                state.model_copy(update={"flags": {**state.flags, "potion_slot_indices": bad}})
            )


def test_source_shop_purchase_fills_first_hole_and_keeps_remaining_indices():
    state = _source().model_copy(
        update={
            "phase": RunPhase.SHOP,
            "potions": ("block_potion",),
            "flags": {**_source().flags, "potion_slot_indices": (2,)},
            "shop": ShopState(
                node_id="physical-shop",
                items=(
                    ShopItemState(
                        slot_id="shop:0",
                        item_id="strength_potion",
                        kind="potion",
                        price=5,
                        base_price=5,
                    ),
                ),
            ),
        }
    )
    state = state.model_copy(update={"player": state.player.model_copy(update={"gold": 50})})
    state = step(state, _choose(state, "shop_buy"))
    assert inventory_slots(state) == ("strength_potion", None, "block_potion")
    assert state.player.gold == 45
    _reload(state)


def test_identical_potions_remain_distinct_by_physical_slot():
    state = _source().model_copy(update={"potions": ("blood_potion",) * 3})
    state = step(state, _choose(state, "use_potion", "potion:1"))
    assert inventory_slots(state) == ("blood_potion", None, "blood_potion")
    state = step(_reload(state), _choose(state, "use_potion", "potion:2"))
    assert inventory_slots(state) == ("blood_potion", None, None)
    assert not any(a.payload.get("potion_slot") == "potion:1" for a in legal_actions(state))


def test_procurement_removes_belt_dexterity_at_the_source_after_procure_boundary():
    state = _fight((), ())
    state = state.model_copy(
        update={
            "relics": ("belt_buckle",),
            "combat": state.combat.model_copy(
                update={
                    "player": state.combat.player.model_copy(update={"statuses": {"dexterity": 2}}),
                    "metadata": {**state.combat.metadata, "relic_counters": {"belt_buckle": 1}},
                }
            ),
        }
    )
    state, events = procure_at_slot(state, "fire_potion")
    assert state.combat.player.statuses.get("dexterity", 0) == 0
    assert state.combat.metadata["replay_potions"] == ("fire_potion",)
    assert state.combat.metadata["replay_potion_slot_indices"] == (0,)
    assert events[0].kind == "event_potion_obtained"
    assert any(e.source_id == "belt_buckle" for e in events[1:])
    _reload(state)


def test_source_discard_is_available_before_combat_without_a_pending_offer():
    state = _source().model_copy(update={"potions": ("blood_potion",)})
    state = state.model_copy(
        update={"flags": {k: v for k, v in state.flags.items() if k != "allow_free_potion_discard"}}
    )
    state = step(state, _choose(state, "discard_potion", "potion:0"))
    assert inventory_slots(state) == (None, None, None)


@pytest.mark.parametrize(
    "event_id", ["RANWID_THE_ELDER", "STONE_OF_ALL_TIME", "THE_FUTURE_OF_POTIONS"]
)
def test_source_event_can_remove_guard_blocks_anytime_use_as_well_as_discard(event_id):
    state = _source().model_copy(
        update={
            "phase": RunPhase.EVENT,
            "event": EventState(
                event_id=event_id,
                name=event_id,
                options=(EventOptionState(option_id="leave", title="Leave"),),
            ),
        }
    )
    assert not any(a.type in {"use_potion", "discard_potion"} for a in legal_actions(state))
    # The explicit lock also persists through a child pickup page with another ID.
    state = state.model_copy(
        update={
            "event": state.event.model_copy(update={"event_id": "CHILD"}),
            "flags": {**state.flags, "can_remove_potions": False},
        }
    )
    assert not any(a.type in {"use_potion", "discard_potion"} for a in legal_actions(state))
