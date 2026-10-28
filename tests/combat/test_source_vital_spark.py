"""Stable source VitalSpark/Tainted affliction identity, entry and removal.

Razhuo/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7:
VitalSparkPower, Tainted affliction, CreatureCmd.Kill and AfflictionModel.
"""

from __future__ import annotations

from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play

from sts2sim import legal_actions, step
from sts2sim.engine.source_affliction_hooks import affliction_id


def state_with_sparks(*cards, amounts=(2,), draw=()):
    state = _state(*cards, draw=draw)
    base = state.combat.monsters[0]
    monsters = tuple(
        base.model_copy(update={"monster_id": f"prism{i}", "statuses": {"vital_spark": amount}})
        for i, amount in enumerate(amounts)
    )
    # Keep a non-Prism alive so removal behavior can be observed after a kill.
    monsters += (base.model_copy(update={"monster_id": "survivor"}),)
    return state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": monsters})}
    )


def amount(card):
    return card.custom.get("source_core_affliction", {}).get("amount", 0)


def generated_skill():
    return {
        "card_id": "fresh",
        "type": "skill",
        "cost": 0,
        "target": "self",
        "effects": {"block": 1},
    }


def test_initial_skills_in_every_combat_pile_receive_tainted():
    state = state_with_sparks(_simple("first"), _simple("held"), draw=(_simple("draw"),))
    old = _simple("old")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "discard_pile": (old,),
                    "exhaust_pile": (_simple("exhausted"),),
                }
            )
        }
    )
    state = _play(state, "first")
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        assert all(
            affliction_id(c) == "tainted" and amount(c) == 2 for c in getattr(state.combat, zone)
        )
    assert state.player.statuses["tainted"] == 2


def test_existing_other_affliction_prevents_vital_spark_debuff():
    bound = _simple("bound", custom={"bound": True})
    state = _play(state_with_sparks(bound), "bound")
    assert not state.player.statuses.get("tainted")
    assert affliction_id(state.combat.discard_pile[0]) == "bound"


def test_initial_multiple_sources_stack_but_late_entry_uses_first_source():
    make = _simple("make", effects={"add_card_to_hand": {"card": generated_skill()}})
    state = state_with_sparks(make, _simple("held"), amounts=(2, 3))
    state = _play(state, "make")
    assert amount(next(c for c in state.combat.hand if c.card_id == "held")) == 5
    fresh = next(c for c in state.combat.hand if c.card_id == "fresh")
    assert amount(fresh) == 2
    assert state.player.statuses["tainted"] == 5
    state = _play(_reload(state), fresh.instance_id)
    assert state.player.statuses["tainted"] == 10


def test_removing_one_source_clears_old_afflictions_even_with_another_prism_alive():
    kill = _simple("kill", card_type="attack", target="enemy", effects={"damage": 100})
    state = state_with_sparks(kill, _simple("held"), amounts=(2, 3))
    action = next(
        a for a in legal_actions(state) if a.card_instance_id == "kill" and a.target_id == "prism0"
    )
    state = step(state, action)
    assert affliction_id(state.combat.hand[0]) == ""
    state = _play(_reload(state), "held")
    assert not state.player.statuses.get("tainted")


def test_skill_that_kills_prism_clears_its_affliction_before_after_card_played():
    kill = _simple("kill", target="enemy", effects={"damage": 100})
    state = state_with_sparks(kill, amounts=(2, 3))
    action = next(
        a for a in legal_actions(state) if a.card_instance_id == "kill" and a.target_id == "prism0"
    )
    state = step(_reload(state), action)
    assert not state.player.statuses.get("tainted")
    assert affliction_id(state.combat.discard_pile[0]) == ""


def test_new_skill_after_removal_gets_remaining_sources_affliction():
    kill = _simple("kill", card_type="attack", target="enemy", effects={"damage": 100})
    make = _simple("make", effects={"add_card_to_hand": {"card": generated_skill()}})
    state = state_with_sparks(kill, make, amounts=(2, 3))
    action = next(
        a for a in legal_actions(state) if a.card_instance_id == "kill" and a.target_id == "prism0"
    )
    state = step(state, action)
    state = _play(state, "make")
    assert not state.player.statuses.get("tainted")  # This old card was cleared.
    fresh = state.combat.hand[0]
    assert amount(fresh) == 3
    state = _play(_reload(state), fresh.instance_id)
    assert state.player.statuses["tainted"] == 3


def test_vital_spark_amount_change_sets_affliction_display_and_changes_debuff():
    grow = _simple(
        "grow", target="enemy", effects={"apply_status": {"target": "enemy", "vital_spark": 2}}
    )
    state = state_with_sparks(grow, _simple("held"))
    action = next(
        a for a in legal_actions(state) if a.card_instance_id == "grow" and a.target_id == "prism0"
    )
    state = step(state, action)
    assert amount(state.combat.hand[0]) == 4
    assert amount(state.combat.discard_pile[0]) == 4
    assert state.player.statuses["tainted"] == 4


def test_artifact_blocks_each_source_application_separately():
    state = state_with_sparks(_simple("skill"), amounts=(2, 3))
    player = state.player.model_copy(update={"statuses": {"artifact": 1}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _play(state, "skill")
    assert state.player.statuses["tainted"] == 3
    assert not state.player.statuses.get("artifact")


def test_combat_start_afflicts_deck_before_opening_hand_draws():
    from combat.test_named_monster_scripts import _combat

    state = _combat("INFESTED_PRISM", statuses=(("VITAL_SPARK", 2),))
    assert all(affliction_id(c) == "tainted" for c in state.combat.hand + state.combat.draw_pile)
    assert not any(affliction_id(c) for c in state.master_deck)


def test_unplayable_skill_is_not_eligible_for_tainted():
    state = state_with_sparks(_simple("first"), _simple("locked", custom={"unplayable": True}))
    state = _play(state, "first")
    assert affliction_id(state.combat.hand[0]) == ""


def test_chains_of_binding_cannot_add_second_affliction_over_tainted():
    state = state_with_sparks(_simple("draw", effects={"draw": 1}), draw=(_simple("drawn"),))
    player = state.player.model_copy(update={"statuses": {"chains_of_binding": 1}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _play(state, "draw")
    assert affliction_id(state.combat.hand[0]) == "tainted"
    assert not state.combat.hand[0].custom.get("bound")


def test_potion_kill_clears_afflictions_before_returning_public_state():
    state = state_with_sparks(_simple("first"), _simple("held"), amounts=(2, 3))
    state = _play(state, "first")
    monsters = (state.combat.monsters[0].model_copy(update={"hp": 1}),) + state.combat.monsters[1:]
    state = state.model_copy(
        update={
            "potions": ("fire_potion",),
            "combat": state.combat.model_copy(update={"monsters": monsters}),
        }
    )
    action = next(
        a for a in legal_actions(state) if a.type == "use_potion" and a.target_id == "prism0"
    )
    state = step(_reload(state), action)
    assert affliction_id(state.combat.hand[0]) == ""
    assert (
        state.player.statuses["tainted"] == 5
    )  # Removing card affliction does not remove existing debuff.
