from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import CombatState, MonsterState, OrbState, RunPhase
from sts2sim.engine.transitions import _card_from_spec

SOURCE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "card_status_runtime_source.json").read_text()
)


def _skill(card_id, effects, **extra):
    return {
        "id": card_id,
        "type": "Skill",
        "target": "Self",
        "cost": 0,
        "effects": effects,
        **extra,
    }


def _attack(card_id="hit", damage=10, **extra):
    return {
        "id": card_id,
        "type": "Attack",
        "target": "Enemy",
        "cost": 0,
        "effects": {"damage": damage},
        **extra,
    }


def _debuff(card_id, status, amount=1):
    return _skill(card_id, {"apply_status": {"target": "enemy", status: amount}}, target="Enemy")


def _fight(*specs, statuses=None, enemy_statuses=None, block=0, relics=()):
    state = new_run(seed=93, character_id="TEST", ascension=0, source_data={"deck": []})
    player = state.player.model_copy(
        update={"energy": 50, "statuses": statuses or {}, "resources": {"star": 20}}
    )
    combat = CombatState(
        player=player,
        hand=tuple(_card_from_spec(spec, i + 1) for i, spec in enumerate(specs)),
        monsters=(
            MonsterState(
                monster_id="a",
                hp=500,
                max_hp=500,
                block=block,
                statuses=enemy_statuses or {},
                intent="Attack",
                intent_damage=10,
            ),
            MonsterState(monster_id="b", hp=500, max_hp=500, intent="Attack", intent_damage=10),
        ),
        draw_per_turn=0,
    )
    return state.model_copy(
        update={
            "phase": RunPhase.COMBAT,
            "player": player,
            "combat": combat,
            "relics": tuple(relics),
        }
    )


def _play(state, card_id, target="a"):
    card = next(c for c in state.combat.hand if c.card_id == card_id.lower())
    action = next(
        a
        for a in legal_actions(state)
        if a.type == "play_card"
        and a.card_instance_id == card.instance_id
        and a.target_id in {None, "player", target}
    )
    return step(state, action)


def _end(state):
    return step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))


@pytest.mark.parametrize(
    "card_id,status,amount",
    [
        ("ACCELERANT", "poison_extra_triggers", 1),
        ("BLACK_HOLE", "blackhole", 3),
        ("CRUELTY", "cruelty", 25),
        ("ENVENOM", "envenom", 1),
        ("OUTBREAK", "outbreak", 11),
        ("SLEIGHT_OF_FLESH", "sleightofflesh", 9),
        ("SMOKESTACK", "smokestack", 5),
        ("SNEAKY", "sneaky", 1),
        ("THUNDER", "thunder", 6),
    ],
)
@pytest.mark.parametrize("upgraded", [False, True])
def test_printed_source_power_installs_without_immediate_trigger(card_id, status, amount, upgraded):
    deltas = {
        "ACCELERANT": 1,
        "BLACK_HOLE": 1,
        "CRUELTY": 25,
        "ENVENOM": 1,
        "OUTBREAK": 4,
        "SLEIGHT_OF_FLESH": 4,
        "SMOKESTACK": 2,
        "SNEAKY": 1,
        "THUNDER": 2,
    }
    amount += deltas[card_id] if upgraded else 0
    state = _play(_fight({**SOURCE[card_id], "upgraded": upgraded}), card_id)
    assert state.player.statuses[status] == amount
    assert [m.hp for m in state.combat.monsters] == [500, 500]


def test_accelerant_repeats_poison_and_decrements_each_trigger_after_reload():
    state = _play(_fight(SOURCE["ACCELERANT"], enemy_statuses={"poison": 5}), "accelerant")
    state = _end(load_state(serialize(state)))
    assert state.combat.monsters[0].hp == 491  # 5 then 4, not 5 twice.
    assert state.combat.monsters[0].statuses["poison"] == 3
    assert sum(e.kind == "monster_poison_damage" for e in state.combat.last_events) == 2


def test_accelerant_stops_poison_when_enemy_dies():
    state = _fight(statuses={"poison_extra_triggers": 2}, enemy_statuses={"poison": 600})
    state = _end(state)
    assert state.combat.monsters[0].hp == 0
    assert sum(e.kind == "monster_poison_damage" for e in state.combat.last_events) == 1


def test_envenom_counts_unblocked_attack_hits_and_artifact_blocks_poison():
    multi = _attack("multi", effects=[{"damage": 4}] * 3)
    state = _fight(SOURCE["ENVENOM"], multi, block=4, enemy_statuses={"artifact": 1})
    state = _play(_play(state, "envenom"), "multi")
    assert state.combat.monsters[0].hp == 492
    assert state.combat.monsters[0].statuses.get("poison") == 1
    assert state.combat.monsters[0].statuses.get("artifact", 0) == 0
    assert (
        sum(
            e.kind == "status_applied" and e.metadata.get("rider") == "envenom"
            for e in state.combat.last_events
        )
        == 1
    )


def test_outbreak_counts_applications_not_poison_amount_or_relic_bonus():
    state = _fight(
        SOURCE["OUTBREAK"],
        *(_debuff(f"p{i}", "poison", 4) for i in range(3)),
        relics=("snecko_skull",),
    )
    state = _play(state, "outbreak")
    for i in range(2):
        state = _play(state, f"p{i}")
    assert [m.hp for m in state.combat.monsters] == [500, 500]
    state = _play(load_state(serialize(state)), "p2")
    assert [m.hp for m in state.combat.monsters] == [489, 489]
    assert state.combat.monsters[0].statuses["poison"] == 15
    assert state.combat.metadata["outbreak_poison_applications"] == 0


def test_blocked_poison_does_not_count_toward_outbreak():
    state = _fight(
        _debuff("poison", "poison", 50), statuses={"outbreak": 11}, enemy_statuses={"artifact": 1}
    )
    state = _play(state, "poison")
    assert state.combat.metadata.get("outbreak_poison_applications", 0) == 0


def test_sleight_only_successful_debuffs_and_not_buffs():
    state = _fight(
        SOURCE["SLEIGHT_OF_FLESH"],
        _debuff("first", "weak"),
        _debuff("buff", "strength", 2),
        _debuff("second", "vulnerable"),
        enemy_statuses={"artifact": 1},
    )
    state = _play(state, "sleight_of_flesh")
    state = _play(_play(state, "first"), "buff")
    assert state.combat.monsters[0].hp == 500
    state = _play(load_state(serialize(state)), "second")
    assert state.combat.monsters[0].hp == 491


@pytest.mark.parametrize("relics,damage", [((), 25), (("paper_phrog",), 30)])
def test_cruelty_and_debilitate_combine_vulnerable_bonus(relics, damage):
    state = _fight(
        SOURCE["CRUELTY"],
        SOURCE["DEBILITATE"],
        _attack(),
        enemy_statuses={"vulnerable": 5},
        relics=relics,
    )
    state = _play(_play(state, "cruelty"), "debilitate")
    hp = state.combat.monsters[0].hp
    state = _play(load_state(serialize(state)), "hit")
    assert hp - state.combat.monsters[0].hp == damage
    assert state.combat.monsters[0].statuses["debilitate"] == 2


@pytest.mark.parametrize("relics,damage", [((), 15), (("paper_krane",), 12)])
def test_debilitate_doubles_weak_reduction_and_ticks_duration(relics, damage):
    state = _fight(SOURCE["DEBILITATE"], enemy_statuses={"weak": 5}, relics=relics)
    state = _play(state, "debilitate")
    hp = state.player.hp
    state = _end(state)
    assert hp - state.player.hp == damage
    assert state.combat.monsters[0].statuses["debilitate"] == 1
    state = _end(state)
    assert "debilitate" not in state.combat.monsters[0].statuses


@pytest.mark.parametrize("source", ["FLANKING", "KNOCKDOWN"])
def test_other_player_multiplier_preserves_owner_across_reload_and_expires(source):
    state = _fight(SOURCE[source], _attack("own"), _attack("ally", custom={"owner_id": "friend"}))
    state = _play(state, source)
    hp = state.combat.monsters[0].hp
    state = _play(state, "own")
    assert hp - state.combat.monsters[0].hp == 10
    hp = state.combat.monsters[0].hp
    state = _play(load_state(serialize(state)), "ally")
    assert hp - state.combat.monsters[0].hp == 20
    state = _end(state)
    assert source.lower() not in state.combat.monsters[0].statuses
    assert f"{source.lower()}_applications" not in state.combat.monsters[0].metadata


def test_flanking_applications_stack_multiplicatively():
    state = _fight(
        SOURCE["FLANKING"],
        {**SOURCE["FLANKING"], "id": "flanking_two"},
        _attack(custom={"owner_id": "friend"}),
    )
    state = _play(_play(state, "flanking"), "flanking_two")
    state = _play(state, "hit")
    assert state.combat.monsters[0].hp == 460


def test_sneaky_triggers_once_for_other_players_attack_not_each_hit_or_own_attack():
    state = _fight(
        SOURCE["SNEAKY"],
        _attack("own"),
        _attack("ally", effects=[{"damage": 2}] * 3, custom={"owner_id": "friend"}),
    )
    state = _play(_play(state, "sneaky"), "own")
    assert state.player.block == 0
    state = _play(load_state(serialize(state)), "ally")
    assert state.player.block == 1


def test_thunder_only_triggers_for_lightning_evokes_and_is_separate_damage():
    state = _fight(SOURCE["THUNDER"], _skill("evoke", {"evoke_orb": {"amount": 1}}))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orb_slots": 3,
                    "orbs": (OrbState(orb_id="lightning"),),
                }
            )
        }
    )
    state = _play(_play(state, "thunder"), "evoke")
    assert sorted(m.hp for m in state.combat.monsters) == [486, 500]
    assert [e.amount for e in state.combat.last_events if e.kind == "monster_damaged"] == [8, 6]


@pytest.mark.parametrize("orb,damage", [("lightning", 4), ("frost", 0)])
def test_thunder_handles_blocked_lightning_hit_but_ignores_other_orbs(orb, damage):
    state = _fight(_skill("evoke", {"evoke_orb": {"amount": 1}}), statuses={"thunder": 6})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orb_slots": 3,
                    "orbs": (OrbState(orb_id=orb),),
                    "monsters": tuple(
                        m.model_copy(update={"block": 10}) for m in state.combat.monsters
                    ),
                }
            )
        }
    )
    state = _play(state, "evoke")
    assert sum(500 - m.hp for m in state.combat.monsters) == damage


def test_black_hole_once_per_gain_or_spend_instance_after_card_damage_and_reload():
    spender = _attack("spend", damage=7, star_cost=3)
    gain = _skill("gain", {"player_resource": {"resource": "star", "amount": 5}})
    state = _fight(SOURCE["BLACK_HOLE"], gain, spender)
    state = _play(_play(state, "black_hole"), "gain")
    assert [m.hp for m in state.combat.monsters] == [497, 497]
    state = _play(load_state(serialize(state)), "spend")
    assert [m.hp for m in state.combat.monsters] == [487, 494]
    damages = [
        (e.source_id, e.amount) for e in state.combat.last_events if e.kind == "monster_damaged"
    ]
    assert damages[0][1] == 7
    assert damages[1:] == [("black_hole", 3), ("black_hole", 3)]


def test_smokestack_created_status_counts_each_card_not_draw_or_move():
    create = _skill(
        "create", {"add_card_to_hand": {"card": {"card_id": "burn", "type": "Status"}, "count": 2}}
    )
    state = _fight(SOURCE["SMOKESTACK"], create)
    state = _play(load_state(serialize(_play(state, "smokestack"))), "create")
    assert [m.hp for m in state.combat.monsters] == [490, 490]
    assert sum(c.card_id == "burn" for c in state.combat.hand) == 2
    assert len(state.combat.metadata["runtime_created_card_ids"]) == 2


def test_black_hole_waits_for_pending_choice_and_survives_reload():
    card = _skill("choose", {"discard_choice": 1}, star_cost=2)
    state = _fight(card, _attack(), statuses={"blackhole": 3})
    state = _play(state, "choose")
    assert state.combat.pending_choices
    assert [m.hp for m in state.combat.monsters] == [500, 500]
    state = load_state(serialize(state))
    state = step(state, next(a for a in legal_actions(state) if a.type == "discard_card"))
    assert [m.hp for m in state.combat.monsters] == [497, 497]
    assert "pending_black_hole" not in state.combat.metadata


def test_successful_self_doom_dispatches_public_callback_and_records_turn():
    listener = _skill(
        "listener",
        {
            "combat_trigger": {
                "trigger": "status_applied",
                "effects": {"block": 2},
            }
        },
    )
    state = _fight(listener, _skill("doom", {"apply_status": {"target": "self", "doom": 2}}))
    state = _play(load_state(serialize(_play(state, "listener"))), "doom")
    assert state.combat.metadata["last_doom_applied_turn"] == state.combat.turn
    assert state.player.statuses["doom"] == 2
    assert state.player.block == 2


def test_imported_other_player_cards_do_not_trigger_local_owned_poison_powers():
    poison = _debuff("poison", "poison", 4)
    poison["custom"] = {"owner_id": "friend"}
    state = _fight(
        poison,
        _attack(custom={"owner_id": "friend"}),
        statuses={"envenom": 2, "outbreak": 11, "sleightofflesh": 9},
    )
    state = _play(_play(state, "poison"), "hit")
    assert state.combat.monsters[0].hp == 490
    assert state.combat.monsters[0].statuses["poison"] == 4
    assert state.combat.metadata.get("outbreak_poison_applications", 0) == 0
