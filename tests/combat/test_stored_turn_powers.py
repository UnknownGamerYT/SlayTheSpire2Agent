from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_combat_engine_monsters import _enter_knowledge_demon_combat

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardInstance, CombatState, MonsterState, OrbState, PlayerState, RunState
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import _card_from_spec


def _source_card(card_id: str, *, upgraded: bool = False) -> CardInstance:
    rows = json.loads((Path(__file__).parents[2] / "data/cache/eng/cards.json").read_text())
    if isinstance(rows, dict):
        rows = rows.get("items", rows.get("data", rows))
    spec = dict(next(row for row in rows if row["id"] == card_id.upper()))
    return _card_from_spec({**spec, "instance_id": card_id, "upgraded": upgraded}, 1)


def _fight(
    *card_ids: str,
    upgraded: bool = False,
    orbs: tuple[str, ...] = (),
    statuses: dict[str, int] | None = None,
    enemy_damage: int = 0,
    monster_count: int = 1,
) -> RunState:
    hand = tuple(_source_card(card_id, upgraded=upgraded) for card_id in card_ids)
    fillers = tuple(CardInstance(
        instance_id=f"filler:{i}", card_id="filler", type="skill", cost=0, target="self",
        effects={"block": 5}, custom={"retain": True},
    ) for i in range(15))
    player = PlayerState(hp=200, max_hp=200, energy=20, max_energy=20, statuses=statuses or {})
    source = {
        "id": "POWER_DUMMY", "name": "Power Dummy", "type": "Normal",
        "min_hp": 1000, "max_hp": 1000,
        "moves": ({"id": "WAIT", "name": "Wait", "intent": "Attack" if enemy_damage else "Buff",
                   "damage": {"normal": enemy_damage}},),
        "attack_pattern": {"initial_move": "WAIT", "states": (
            {"id": "WAIT", "move_id": "WAIT", "type": "move", "next": "WAIT"},
        )},
    }
    monsters = tuple(MonsterState(
        monster_id=f"dummy:{i}", hp=1000, max_hp=1000, move_id="WAIT",
        metadata={"source_monster_id": "POWER_DUMMY"},
    ) for i in range(monster_count))
    return RunState(
        seed=9, character_id="TEST", rng=new_rng_state(9), player=player,
        master_deck=hand + fillers, flags={"monsters": (source,)},
        combat=CombatState(player=player, hand=hand, draw_pile=fillers,
                           draw_per_turn=5, monsters=monsters,
                           orb_slots=3, orbs=tuple(OrbState(orb_id=orb) for orb in orbs)),
    )


def _play(state: RunState, card_id: str) -> RunState:
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    return step(state, next(a for a in legal_actions(state)
                           if a.type == "play_card" and a.card_instance_id == card.instance_id))


def _end(state: RunState) -> RunState:
    return step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))


def _knowledge_choice(card_id: str, *, artifact: bool = False) -> RunState:
    state = _enter_knowledge_demon_combat()
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "metadata": {**state.combat.metadata,
                     "knowledge_demon_curse_counter": 2 if card_id == "waste_away" else 0},
        "player": state.player.model_copy(update={
            "statuses": {"artifact": 1} if artifact else {},
        }),
    })})
    state = _end(state)
    candidates = state.combat.pending_choices[0].metadata["generated_cards"]
    chosen = next(card["instance_id"] for card in candidates if card["card_id"] == card_id)
    return step(state, next(a for a in legal_actions(state) if a.card_instance_id == chosen))


@pytest.mark.parametrize("upgraded,initial", [(False, 4), (True, 5)])
def test_biased_cognition_loses_focus_each_turn_including_below_zero_after_reload(
    upgraded: bool, initial: int,
) -> None:
    state = _play(_fight("biased_cognition", upgraded=upgraded), "biased_cognition")
    assert state.player.statuses["focus"] == initial
    for turn in range(1, initial + 2):
        state = _end(load_state(serialize(state)))
        assert state.player.statuses.get("focus", 0) == initial - turn
    # Negative Focus changes actual orb output, rather than only its display.
    assert state.combat is not None
    evoke = CardInstance(instance_id="evoke", card_id="evoke", type="skill", cost=0,
                         target="self", effects={"evoke_orb": 1})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (evoke,), "orbs": (OrbState(orb_id="frost"),),
    })})
    state = _play(state, "evoke")
    assert state.player.block == 4  # Frost's base evoke 5, Focus -1.


@pytest.mark.parametrize("upgraded,intangible", [(False, 2), (True, 3)])
def test_wraith_form_loses_dexterity_persistently_and_changes_actual_skill_block(
    upgraded: bool, intangible: int,
) -> None:
    state = _play(_fight("wraith_form", upgraded=upgraded), "wraith_form")
    assert state.player.statuses["intangible"] == intangible
    assert state.player.statuses.get("dexterity", 0) == 0
    state = _end(load_state(serialize(state)))
    state = _end(state)
    assert state.player.statuses["dexterity"] == -2
    state = _play(state, "filler")
    assert state.player.block == 3


@pytest.mark.parametrize("card_id", ["biased_cognition", "wraith_form", "neurosurge"])
def test_artifact_blocks_stored_debuff_but_keeps_its_immediate_benefit(card_id: str) -> None:
    state = _play(_fight(card_id, statuses={"artifact": 1}), card_id)
    assert card_id.replace("_", "") not in state.player.statuses
    assert "artifact" not in state.player.statuses
    if card_id == "biased_cognition":
        assert state.player.statuses["focus"] == 4
    if card_id == "wraith_form":
        assert state.player.statuses["intangible"] == 2
    state = _end(state)
    assert state.player.statuses.get("dexterity", 0) == 0
    assert state.player.statuses.get("doom", 0) == 0


@pytest.mark.parametrize("upgraded,amount", [(False, 2), (True, 3)])
def test_coolant_waits_until_next_start_and_counts_unique_orbs(upgraded: bool, amount: int) -> None:
    state = _play(_fight("coolant", upgraded=upgraded, orbs=("dark", "dark", "plasma")), "coolant")
    assert state.player.block == 0
    for _ in range(2):
        state = _end(load_state(serialize(state)))
        assert state.player.block == amount * 2


@pytest.mark.parametrize("upgraded,amount", [(False, 6), (True, 9)])
def test_countdown_targets_one_random_living_enemy_each_start_and_replays(
    upgraded: bool, amount: int,
) -> None:
    state = _play(_fight("countdown", upgraded=upgraded, monster_count=3), "countdown")
    assert all(not m.statuses.get("doom") for m in state.combat.monsters)
    restored = load_state(serialize(state))
    for turn in range(1, 4):
        state, restored = _end(state), _end(restored)
        assert serialize(state) == serialize(restored)
        assert sum(m.statuses.get("doom", 0) for m in state.combat.monsters) == turn * amount


def test_countdown_artifact_blocks_one_future_application_without_consuming_the_power() -> None:
    state = _fight("countdown")
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (state.combat.monsters[0].model_copy(update={"statuses": {"artifact": 1}}),),
    })})
    state = _end(_play(state, "countdown"))
    assert state.combat.monsters[0].statuses.get("doom", 0) == 0
    assert "artifact" not in state.combat.monsters[0].statuses
    state = _end(load_state(serialize(state)))
    assert state.combat.monsters[0].statuses["doom"] == 6


@pytest.mark.parametrize("upgraded,amount", [(False, 8), (True, 10)])
def test_crimson_mantle_pays_hp_and_grants_block_only_on_future_turns(upgraded, amount) -> None:
    state = _play(_fight("crimson_mantle", upgraded=upgraded), "crimson_mantle")
    assert state.player.hp == 200 and state.player.block == 0
    for turn in (1, 2):
        state = _end(load_state(serialize(state)))
        assert state.player.hp == 200 - turn
        assert state.player.block == amount
        assert state.combat.metadata["hp_loss_count_this_turn"] == 1


def test_crimson_mantle_copies_stack_both_block_and_self_damage_and_keep_counters() -> None:
    state = _fight("crimson_mantle")
    second = state.combat.hand[0].model_copy(update={"instance_id": "second_mantle"})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (*state.combat.hand, second),
    })})
    state = _play(_play(state, "crimson_mantle"), "crimson_mantle")
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 198 and state.player.block == 16


@pytest.mark.parametrize("upgraded,amount", [(False, 6), (True, 9)])
def test_inferno_only_reacts_to_own_turn_actual_hp_loss_including_its_start_cost(upgraded, amount):
    state = _play(_fight("inferno", upgraded=upgraded, enemy_damage=4), "inferno")
    assert state.player.hp == 200 and state.combat.monsters[0].hp == 1000
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 195
    assert state.combat.monsters[0].hp == 1000 - amount  # Enemy hit does not trigger it.
    self_loss = CardInstance(instance_id="loss", card_id="loss", type="skill", cost=0,
                             target="self", effects={"hp_loss": 3})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (self_loss,), "player": state.player.model_copy(update={"statuses": {
            **state.player.statuses, "rupture": 2,
        }}),
    })})
    state = _play(state, "loss")
    assert state.player.hp == 192
    assert state.combat.monsters[0].hp == 1000 - amount * 2
    assert state.player.statuses["strength"] == 2
    assert state.combat.metadata["hp_loss_count_this_turn"] == 2


def test_buffer_prevents_inferno_rupture_and_loss_counters_for_a_prevented_start_cost() -> None:
    state = _play(_fight("inferno", statuses={"buffer": 1, "rupture": 1}), "inferno")
    state = _end(state)
    assert state.player.hp == 200
    assert state.combat.monsters[0].hp == 1000
    assert state.player.statuses.get("strength", 0) == 0
    assert state.combat.metadata.get("hp_loss_count_this_turn", 0) == 0


def test_inferno_and_rupture_react_to_end_hand_hp_loss_before_enemy_turn() -> None:
    state = _play(_fight("inferno", statuses={"rupture": 1}), "inferno")
    burn = _source_card("burn")
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (burn,)})})
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 197  # Burn2, then Inferno's next-start HP1.
    assert state.combat.monsters[0].hp == 988
    assert state.player.statuses["strength"] == 2
    assert state.combat.metadata["hp_loss_count_this_combat"] == 2


def test_inferno_and_rupture_react_once_to_player_turn_retaliation_damage() -> None:
    state = _fight("inferno", "coolant", statuses={"rupture": 1})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (state.combat.monsters[0].model_copy(update={"statuses": {"galvanic": 2}}),),
    })})
    state = _play(state, "inferno")
    assert state.player.hp == 198 and state.combat.monsters[0].hp == 994
    state = _play(load_state(serialize(state)), "coolant")
    assert state.player.hp == 196 and state.combat.monsters[0].hp == 988
    assert state.player.statuses["strength"] == 2
    assert state.combat.metadata["hp_loss_count_this_combat"] == 2


@pytest.mark.parametrize("upgraded,energy", [(False, 3), (True, 4)])
def test_neurosurge_gives_energy_and_draw_now_but_doom_only_on_following_turns(upgraded, energy):
    state = _fight("neurosurge", upgraded=upgraded)
    cost = state.combat.hand[0].cost
    state = _play(state, "neurosurge")
    assert state.player.energy == 20 - cost + energy
    assert len(state.combat.hand) == 2
    assert state.player.statuses.get("doom", 0) == 0
    for turn in (1, 2):
        state = _end(load_state(serialize(state)))
        assert state.player.statuses["doom"] == turn * 3


@pytest.mark.parametrize("upgraded,amount", [(False, 4), (True, 6)])
def test_prep_time_adds_vigor_each_turn_without_granting_any_on_play(upgraded, amount):
    state = _play(_fight("prep_time", upgraded=upgraded), "prep_time")
    assert state.player.statuses.get("vigor", 0) == 0
    for turn in (1, 2):
        state = _end(load_state(serialize(state)))
        assert state.player.statuses["vigor"] == amount * turn


@pytest.mark.parametrize("upgraded,initial", [(False, 5), (True, 10)])
def test_rolling_boulder_deals_and_increases_damage_at_each_start_after_reload(upgraded, initial):
    state = _play(_fight("rolling_boulder", upgraded=upgraded), "rolling_boulder")
    assert state.combat.monsters[0].hp == 1000
    total = 0
    for turn in range(3):
        state = _end(load_state(serialize(state)))
        total += initial + 5 * turn
        assert state.combat.monsters[0].hp == 1000 - total
        assert state.player.statuses["rollingboulder"] == initial + 5 * (turn + 1)


def test_sentry_mode_creates_real_sweeping_gaze_only_at_turn_start_with_unique_ids() -> None:
    state = _play(_fight("sentry_mode"), "sentry_mode")
    assert not any(card.card_id == "sweeping_gaze" for card in state.combat.hand)
    all_ids: list[str] = []
    for _ in range(3):
        state = _end(load_state(serialize(state)))
        new = [event.target_id for event in state.combat.last_events
               if event.kind == "card_added_to_hand" and event.source_id == "sentry_mode"]
        assert len(new) == 1
        all_ids.extend(new)
        gaze = next(card for card in state.combat.hand if card.instance_id == new[0])
        assert gaze.card_id == "sweeping_gaze" and gaze.exhausts
        assert gaze.effects == _source_card("sweeping_gaze").effects
    assert len(set(all_ids)) == 3


@pytest.mark.parametrize("upgraded,damage", [(False, 6), (True, 8)])
@pytest.mark.parametrize("orb", ["frost", "dark"])
def test_hailstorm_only_triggers_at_end_with_a_frost_orb(upgraded, damage, orb):
    state = _play(_fight("hailstorm", upgraded=upgraded, orbs=(orb,)), "hailstorm")
    assert state.combat.monsters[0].hp == 1000
    state = _end(load_state(serialize(state)))
    assert state.combat.monsters[0].hp == 1000 - (damage if orb == "frost" else 0)
    assert state.player.statuses["hailstorm"] == damage


@pytest.mark.parametrize("card_id", ["mind_rot", "waste_away"])
def test_draw_and_energy_debuffs_apply_only_to_following_turns_and_survive_reload(card_id):
    state = _knowledge_choice(card_id)
    assert state.combat.turn == 2
    assert not state.combat.pending_choices
    for _ in range(2):
        draws = [event for event in state.combat.last_events if event.kind == "cards_drawn"]
        assert sum(event.amount or 0 for event in draws) == (4 if card_id == "mind_rot" else 5)
        assert state.player.energy == (2 if card_id == "waste_away" else 3)
        state = _end(load_state(serialize(state)))


@pytest.mark.parametrize("card_id", ["mind_rot", "waste_away"])
def test_artifact_can_block_knowledge_demon_draw_and_energy_curse(card_id):
    state = _knowledge_choice(card_id, artifact=True)
    assert "artifact" not in state.player.statuses
    assert card_id not in state.player.statuses
    assert len(state.combat.hand) == 5
    assert state.player.energy == 3
