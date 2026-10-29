from __future__ import annotations

from test_named_monster_scripts import _combat
from test_remaining_monster_scripts import _end, _play

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine.transitions import _card_from_spec


def _play_actions(state):
    return [a for a in legal_actions(state) if a.type == "play_card"]


def _burrow_fight():
    state = _combat("TUNNELER", moves=(
        {"id": "BURROW", "intent": "Defend + Buff", "block": 32,
         "powers": ({"power_id": "BURROWED", "amount": 1},)},
        {"id": "BELOW", "intent": "Attack", "damage": {"normal": 23}},
        {"id": "DIZZY", "intent": "Stun"},
        {"id": "BITE", "intent": "Attack", "damage": {"normal": 13}},
    ))
    source = dict(state.flags["monsters"][0])
    source["attack_pattern"] = {"initial_move": "BURROW", "states": tuple({
        "id": f"{move}_MOVE", "move_id": move, "next": f"{following}_MOVE",
    } for move, following in (
        ("BURROW", "BELOW"), ("BELOW", "BELOW"), ("DIZZY", "BITE"), ("BITE", "BURROW"),
    ))}
    return state.model_copy(update={"flags": {**state.flags, "monsters": (source,)}})


def test_burrowed_retains_block_and_block_break_changes_intent_to_dizzy() -> None:
    state = _end(_burrow_fight())
    assert state.combat.monsters[0].block == 32
    assert state.combat.monsters[0].move_id == "BELOW"
    state = _end(state)
    assert state.combat.monsters[0].block == 32
    assert state.combat.monsters[0].move_id == "BELOW"
    state = _play(state, effects={"damage": 32})
    monster = state.combat.monsters[0]
    assert monster.hp == 100
    assert monster.block == 0
    assert "burrowed" not in monster.statuses
    assert monster.move_id == "DIZZY"
    assert monster.intent_damage == 0
    before = state.combat.player.hp
    state = load_state(serialize(state))
    state = _end(state)
    assert state.combat.player.hp == before
    assert state.combat.monsters[0].move_id == "BITE"


def test_constrict_is_blockable_and_removed_when_its_applier_dies() -> None:
    state = _combat("SLITHERING_STRANGLER", moves=(
        {"id": "CONSTRICT", "intent": "Debuff",
         "powers": ({"power_id": "CONSTRICT", "amount": 3, "target": "player"},)},
        {"id": "WAIT", "intent": "Unknown"},
    ))
    state = _end(state)
    assert state.combat.player.statuses["constrict"] == 3
    before = state.combat.player.hp
    state = _play(state, card_type="skill", effects={"block": 2})
    state = _end(state)
    assert state.combat.player.hp == before - 1
    state = _play(state, effects={"damage": 999})
    assert "constrict" not in state.player.statuses


def test_artifact_blocks_constrict_without_creating_a_linked_drain() -> None:
    state = _combat("SLITHERING_STRANGLER", moves=({
        "id": "CONSTRICT", "intent": "Debuff",
        "powers": ({"power_id": "CONSTRICT", "amount": 3, "target": "player"},),
    },))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "player": state.combat.player.model_copy(update={"statuses": {"artifact": 1}}),
    })})
    state = _end(state)
    assert "constrict" not in state.combat.player.statuses
    assert "constrict_linked_amount" not in state.combat.monsters[0].metadata


def test_constrict_only_removes_the_dead_appliers_share() -> None:
    state = _combat("SLITHERING_STRANGLER", moves=(
        {"id": "CONSTRICT", "intent": "Debuff",
         "powers": ({"power_id": "CONSTRICT", "amount": 3, "target": "player"},)},
        {"id": "WAIT", "intent": "Unknown"},
    ))
    first = state.combat.monsters[0]
    second = first.model_copy(update={"monster_id": "strangler_2"})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, second),
    })})
    state = _end(state)
    assert state.combat.player.statuses["constrict"] == 6
    state = _play(state, effects={"damage": 999}, target=first.monster_id)
    assert state.combat.player.statuses["constrict"] == 3
    before = state.combat.player.hp
    state = _end(state)
    assert state.combat.player.hp == before - 3


def _chains_fight():
    return _end(_combat("QUEEN", moves=(
        {"id": "PUPPET_STRINGS", "intent": "Debuff", "powers": (
            {"power_id": "CHAINS_OF_BINDING", "amount": 3, "target": "player"},
        )},
        {"id": "WAIT", "intent": "Unknown"},
    )))


def test_chains_bind_first_three_draws_and_only_one_bound_card_can_be_played() -> None:
    state = _chains_fight()
    bound = [c for c in state.combat.hand if c.custom.get("bound")]
    assert len(bound) == 3
    state = step(state, next(a for a in _play_actions(state)
                            if a.card_instance_id == bound[0].instance_id))
    state = load_state(serialize(state))
    bound_ids = {c.instance_id for c in state.combat.hand if c.custom.get("bound")}
    assert bound_ids
    assert all(a.card_instance_id not in bound_ids for a in _play_actions(state))
    assert len(_play_actions(state)) == 2
    state = _end(state)
    assert state.combat.metadata["bound_cards_played_this_turn"] == 0
    assert sum(bool(c.custom.get("bound")) for c in state.combat.hand) == 3
    assert not any(c.custom.get("bound") for c in (
        state.combat.draw_pile + state.combat.discard_pile + state.combat.exhaust_pile
    ))


def test_draws_after_first_three_are_not_newly_bound() -> None:
    state = _chains_fight()
    extra = _card_from_spec({"card_id": "extra", "instance_id": "extra", "type": "skill"}, 99)
    draw = _card_from_spec({
        "card_id": "draw", "instance_id": "draw", "type": "skill", "cost": 0,
        "effects": {"draw": 1}, "target": "self",
    }, 100)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (*state.combat.hand, draw), "draw_pile": (extra,),
    })})
    state = step(state, next(a for a in _play_actions(state) if a.card_instance_id == "draw"))
    drawn = next(c for c in state.combat.hand if c.instance_id == "extra")
    assert not drawn.custom.get("bound")


def test_soar_halves_attacks_but_not_potions_and_ends_when_owl_lands() -> None:
    state = _end(_combat("OWL_MAGISTRATE", moves=(
        {"id": "JUDICIAL_FLIGHT", "intent": "Buff",
         "powers": ({"power_id": "SOAR", "amount": 1},)},
        {"id": "VERDICT", "intent": "Attack", "damage": {"normal": 5}},
    )))
    assert state.combat.monsters[0].statuses["soar"] == 1
    state = _play(state, effects={"damage": 10})
    assert state.combat.monsters[0].hp == 95
    state = state.model_copy(update={"potions": ("fire_potion",)})
    state = step(state, next(a for a in legal_actions(state) if a.type == "use_potion"))
    assert state.combat.monsters[0].hp == 75
    state = _end(state)
    assert "soar" not in state.combat.monsters[0].statuses
    state = _play(state, effects={"damage": 10})
    assert state.combat.monsters[0].hp == 65


def test_ringing_limits_exactly_one_player_turn() -> None:
    state = _end(_combat("CEREMONIAL_BEAST", moves=(
        {"id": "BEAST_CRY", "intent": "Debuff",
         "powers": ({"power_id": "RINGING", "amount": 1, "target": "player"},)},
        {"id": "WAIT", "intent": "Unknown"},
    )))
    assert state.combat.player.statuses["ringing"] == 1
    state = step(state, _play_actions(state)[0])
    assert _play_actions(state) == []
    state = _end(state)
    assert "ringing" not in state.combat.player.statuses
    assert len(_play_actions(state)) >= 2
