"""Stable affliction behavior through public actions and save/reload."""

from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardInstance, CombatState, MonsterState, PlayerState, RunState
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.source_modifier_hooks import affliction_id


def _card(name, kind="skill", *, affliction="", effects=None, cost=0, **custom):
    if affliction:
        custom["source_core_affliction"] = {"id": affliction, "amount": 1}
    return CardInstance(
        instance_id=name,
        card_id=name,
        type=kind,
        cost=cost,
        target="enemy" if kind == "attack" else "self",
        effects=effects or {"block": 1},
        custom={"retain": True, **custom},
    )


def _fight(*cards, statuses=None, enemy_statuses=None, draw=()):
    player = PlayerState(hp=100, max_hp=100, energy=20, max_energy=20, statuses=statuses or {})
    return RunState(
        seed=8,
        character_id="TEST",
        rng=new_rng_state(8),
        player=player,
        master_deck=cards,
        combat=CombatState(
            player=player,
            hand=cards,
            draw_pile=draw,
            draw_per_turn=0,
            monsters=(
                MonsterState(
                    monster_id="target", hp=500, max_hp=500, statuses=enemy_statuses or {}
                ),
            ),
        ),
    )


def _plays(state):
    return {a.card_instance_id: a for a in legal_actions(state) if a.type == "play_card"}


def _play(state, instance):
    return step(load_state(serialize(state)), _plays(state)[instance])


def _end(state):
    return step(load_state(serialize(state)), {"type": "end_turn"})


def _all(state):
    c = state.combat
    return c.hand + c.draw_pile + c.discard_pile + c.exhaust_pile


def test_bound_draw_budget_counts_successful_afflictions_and_respects_occupied_slots():
    state = _fight(
        _card("draw", effects={"draw": 4}),
        statuses={"chains_of_binding": 3},
        draw=(_card("occupied", affliction="tainted"), _card("one"), _card("two"), _card("three")),
    )
    result = _play(state, "draw")
    assert affliction_id(result.combat.hand[0]) == "tainted"
    assert [affliction_id(c) for c in result.combat.hand[1:]] == ["bound"] * 3
    assert result.combat.metadata["chains_cards_drawn_this_turn"] == 3
    result = _play(result, "one")
    assert {"two", "three"}.isdisjoint(_plays(result))


def test_bound_without_chains_power_does_not_limit_play():
    state = _fight(_card("one", affliction="bound"), _card("two", affliction="bound"))
    result = _play(state, "one")
    assert "two" in _plays(result)


def test_tangled_only_changes_entangled_attacks_and_expires_at_next_owner_end():
    state = _fight(
        _card("open", "attack", cost=1),
        _card("occupied", "attack", affliction="galvanized", cost=1),
        statuses={"tangled": 2},
    )
    player = state.player.model_copy(update={"energy": 1})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    assert "open" not in _plays(state)
    assert "occupied" in _plays(state)
    result = _end(state)
    assert "tangled" not in result.player.statuses
    assert "open" in _plays(result)
    assert not any(affliction_id(c) == "entangled" for c in _all(result))


def test_ringing_only_blocks_its_own_affliction_after_any_first_card():
    state = _fight(
        _card("first", affliction="bound"),
        _card("second", affliction="bound"),
        _card("ringing"),
        statuses={"ringing": 1},
    )
    result = _play(state, "first")
    assert "second" in _plays(result)
    assert "ringing" not in _plays(result)
    result = _end(result)
    assert "ringing" not in result.player.statuses
    assert "ringing" in _plays(result)


def test_smog_does_not_overwrite_another_affliction_or_block_that_skill():
    state = _fight(
        _card("first"), _card("occupied", affliction="bound"), _card("smog"), statuses={"smoggy": 1}
    )
    result = _play(state, "first")
    assert "occupied" in _plays(result)
    assert "smog" not in _plays(result)
    assert (
        affliction_id(next(c for c in result.combat.hand if c.instance_id == "occupied")) == "bound"
    )
    result = _end(result)
    assert "smog" in _plays(result)
    assert result.player.statuses["smoggy"] == 1


def test_first_skill_does_not_afflict_existing_nested_skill_before_it_finishes():
    state = _fight(
        _card("first", effects={"block": 2, "play_top_card": 1}),
        statuses={"smoggy": 1},
        draw=(_card("nested", effects={"block": 3}),),
    )
    result = _play(state, "first")
    assert result.player.block == 5
    assert result.combat.metadata["skills_played_this_turn"] == 2


def test_new_skill_created_during_first_skill_is_immediately_smogged():
    state = _fight(
        _card(
            "first",
            effects={
                "add_card_to_hand": {
                    "card_id": "generated",
                    "count": 1,
                    "card": {
                        "card_id": "generated",
                        "type": "skill",
                        "cost": 0,
                        "effects": {"block": 4},
                    },
                }
            },
        ),
        statuses={"smoggy": 1},
    )
    result = _play(state, "first")
    generated = next(c for c in result.combat.hand if c.card_id == "generated")
    assert affliction_id(generated) == "smog"
    assert generated.instance_id not in _plays(result)


@pytest.mark.parametrize("occupied", [False, True])
def test_galvanic_only_damages_a_galvanized_power_after_its_block(occupied):
    card = _card("power", "power", affliction="bound" if occupied else "", effects={"block": 2})
    result = _play(_fight(card, enemy_statuses={"galvanic": 6}), "power")
    assert result.player.hp == (100 if occupied else 96)
    assert result.player.block == (2 if occupied else 0)


def test_hex_only_makes_hexed_cards_ethereal_and_retained_hexed_cards_exhaust():
    state = _fight(_card("hexed"), _card("occupied", affliction="bound"), statuses={"hex": 2})
    result = _end(state)
    assert [c.instance_id for c in result.combat.exhaust_pile] == ["hexed"]
    assert [c.instance_id for c in result.combat.hand] == ["occupied"]


def test_hex_removal_preserves_independent_ethereal_keyword():
    state = _fight(_card("hexed"), _card("ethereal", ethereal=True), statuses={"hex": 2})
    state = _play(state, "hexed")
    player = state.player.model_copy(update={"statuses": {}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    result = _end(state)
    assert [c.instance_id for c in result.combat.exhaust_pile] == ["ethereal"]
    assert not any(affliction_id(c) == "hexed" for c in _all(result))


def test_hex_is_removed_when_its_actual_applier_dies_with_another_enemy_alive():
    state = _fight(
        _card("kill", "attack", affliction="bound", effects={"damage": 999}),
        _card("hexed"),
        statuses={"hex": 2},
    )
    combat = state.combat
    state = state.model_copy(
        update={
            "combat": combat.model_copy(
                update={
                    "monsters": combat.monsters
                    + (MonsterState(monster_id="survivor", hp=100, max_hp=100),),
                    "metadata": {"source_hex_applier": "target"},
                }
            )
        }
    )
    result = step(
        load_state(serialize(state)),
        next(
            action
            for action in legal_actions(state)
            if action.card_instance_id == "kill" and action.target_id == "target"
        ),
    )
    assert "hex" not in result.player.statuses
    assert not any(affliction_id(c) == "hexed" for c in _all(result))
    result = _end(result)
    assert [c.instance_id for c in result.combat.hand] == ["hexed"]


def test_multiple_galvanic_sources_stack_initial_affliction_but_only_first_claims_entry():
    state = _fight(
        _card("initial", "power", affliction="galvanized"),
        _card("trigger"),
        enemy_statuses={"galvanic": 2},
    )
    combat = state.combat
    state = state.model_copy(
        update={
            "combat": combat.model_copy(
                update={
                    "monsters": combat.monsters
                    + (
                        MonsterState(
                            monster_id="second", hp=100, max_hp=100, statuses={"galvanic": 3}
                        ),
                    ),
                }
            )
        }
    )
    state = _play(state, "trigger")
    assert state.combat.hand[0].custom["source_core_affliction"]["amount"] == 6
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": state.combat.hand + (_card("new", "power"), _card("trigger2")),
                }
            )
        }
    )
    state = _play(state, "trigger2")
    assert state.combat.hand[1].custom["source_core_affliction"]["amount"] == 2


@pytest.mark.parametrize("identity,power", [("smog", "smoggy"), ("bound", "chains_of_binding")])
def test_affliction_without_a_power_removal_callback_remains_after_source_removal(identity, power):
    state = _fight(_card("first"), _card("marked", affliction=identity), statuses={power: 1})
    state = _play(state, "first")
    player = state.player.model_copy(update={"statuses": {}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    result = _end(state)
    marked = next(c for c in result.combat.hand if c.instance_id == "marked")
    assert affliction_id(marked) == identity
    assert "marked" in _plays(result)


def test_all_afflictions_accept_unplayable_cards_when_type_allows():
    # AfflictionModel.CanAfflictUnplayableCards defaults to true. Only Tainted
    # narrows the accepted card type, and its lifecycle has a separate fixture.
    state = _fight(
        _card("trigger"), _card("unplayable", "attack", unplayable=True), statuses={"tangled": 1}
    )
    result = _play(state, "trigger")
    assert affliction_id(result.combat.hand[0]) == "entangled"
    assert "unplayable" not in _plays(result)
