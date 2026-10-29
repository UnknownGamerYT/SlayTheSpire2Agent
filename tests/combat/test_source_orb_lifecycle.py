"""Public action regressions for the stable source orb queue and owner hooks."""

from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, step
from sts2sim.engine import OrbState, RunPhase
from sts2sim.engine.rng import STS2_STABLE_DERIVATION, RunRng, capture_random_state


def _orb_state(*cards, orbs=(), relics=(), statuses=None, **kwargs):
    state = _state(*(cards or (_simple("held"),)), relics=relics, statuses=statuses, **kwargs)
    return state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orb_slots": max(3, len(orbs)),
                    "orbs": tuple(
                        OrbState(orb_id=orb, value=4 if orb == "glass" else 6) for orb in orbs
                    ),
                }
            )
        }
    )


def _events(state):
    return state.replay_log[-1].events


def _choose(state, kind, identifier=None):
    return step(
        _reload(state),
        next(
            a
            for a in legal_actions(state)
            if a.type == kind and (identifier is None or a.card_instance_id == identifier)
        ),
    )


def test_automatic_end_passives_follow_queue_order_and_stop_on_combat_end():
    state = _orb_state(orbs=("glass", "lightning", "frost", "dark"))
    monster = state.combat.monsters[0].model_copy(update={"hp": 5})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster,),
                }
            )
        }
    )
    state = _end_turn(_reload(state))
    assert state.phase == RunPhase.REWARD
    assert [(e.source_id, e.amount) for e in _events(state) if e.kind == "monster_damaged"] == [
        ("orb:glass", 4),
        ("orb:lightning", 1),
    ]
    assert not any(e.source_id in {"orb:frost", "orb:dark"} for e in _events(state))


def test_plasma_waits_for_next_turn_and_never_creates_a_queued_energy_power():
    state = _orb_state(orbs=("plasma",), statuses={"well_laid_plans": 1})
    pending = _end_turn(_reload(state))
    assert pending.combat.turn == 1
    assert pending.player.energy == 20
    assert "next_turn_energy" not in pending.player.statuses
    assert not any(e.source_id == "orb:plasma" for e in _events(pending))
    result = _choose(pending, "proceed")
    assert result.player.energy == 21
    assert "next_turn_energy" not in result.player.statuses


@pytest.mark.parametrize("focus,damage", [(2, 24), (-4, 0)])
def test_glass_repeated_evoke_doubles_the_focus_adjusted_passive(focus, damage):
    state = _play(
        _reload(_orb_state(_source("DUALCAST"), orbs=("glass",), statuses={"focus": focus})),
        "dualcast",
    )
    assert state.combat.monsters[0].hp == 100 - damage
    assert not state.combat.orbs


def test_zero_glass_passive_keeps_raw_value_until_focus_recovers():
    state = _end_turn(_reload(_orb_state(orbs=("glass",), statuses={"focus": -4})))
    assert state.combat.orbs[0].value == 4
    assert state.combat.monsters[0].hp == 100
    player = state.player.model_copy(update={"statuses": {}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 96
    assert state.combat.orbs[0].value == 3


def test_zero_lightning_evoke_still_returns_target_for_thunder_without_extra_rng():
    state = _orb_state(
        _source("DUALCAST"), orbs=("lightning",), statuses={"focus": -99, "thunder": 5}
    )
    state = state.model_copy(
        update={
            "character_id": "DEFECT",
            "rng": capture_random_state(
                RunRng("zero-lightning-thunder", derivation=STS2_STABLE_DERIVATION)
            ),
        }
    )
    result = _play(_reload(state), "dualcast")
    assert result.combat.monsters[0].hp == 90
    assert result.rng.streams["combat_targets"].draws == 2
    assert [
        e.amount
        for e in _events(result)
        if e.source_id == "thunder" and e.kind == "monster_damaged"
    ] == [5, 5]
    assert _play(_reload(state), "dualcast") == result


def test_dark_ties_keep_encounter_order_instead_of_sorting_monster_ids():
    state = _orb_state(_source("DUALCAST"), orbs=("dark",))
    monster = state.combat.monsters[0]
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (
                        monster.model_copy(update={"monster_id": "z_first", "hp": 30}),
                        monster.model_copy(update={"monster_id": "a_second", "hp": 30}),
                    ),
                }
            )
        }
    )
    state = _play(_reload(state), "dualcast")
    assert [m.hp for m in state.combat.monsters] == [18, 30]


@pytest.mark.parametrize(
    "card_id,upgraded,block",
    [
        ("DUALCAST", False, 10),
        ("QUADCAST", False, 20),
        ("MULTI_CAST", False, 100),
        ("MULTI_CAST", True, 105),
    ],
)
def test_repeated_evoke_cards_keep_front_orb_until_last_evoke(card_id, upgraded, block):
    state = _orb_state(_source(card_id, upgraded=upgraded), orbs=("frost", "lightning"))
    state = _play(_reload(state), card_id.lower())
    assert state.player.block == block
    assert state.combat.monsters[0].hp == 100
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]


def test_shatter_evokes_each_front_orb_twice_before_moving_to_the_next():
    state = _orb_state(_source("SHATTER"), orbs=("lightning", "frost"))
    state = _play(_reload(state), "shatter")
    assert state.combat.monsters[0].hp == 77  # Card7 and two Lightning evokes8.
    assert state.player.block == 10
    assert not state.combat.orbs
    assert [
        e.metadata["orb"]
        for e in _events(state)
        if e.metadata.get("trigger") == "evoke" and e.amount
    ] == [
        "lightning",
        "lightning",
        "frost",
        "frost",
    ]


@pytest.mark.parametrize("upgraded,charge", [(False, 18), (True, 24)])
def test_loop_uses_front_orb_after_hand_draw_without_installing_text_trigger(upgraded, charge):
    state = _orb_state(
        _source("LOOP", upgraded=upgraded),
        orbs=("dark", "lightning"),
        draw=(_simple("fresh"),),
        draw_per_turn=1,
    )
    state = _play(state, "loop")
    state = _end_turn(_reload(state))
    assert state.combat.orbs[0].value == charge
    assert state.combat.monsters[0].hp == 97
    events = _events(state)
    assert next(i for i, e in enumerate(events) if e.kind == "card_drawn") < next(
        i for i, e in enumerate(events) if e.source_id == "loop"
    )
    assert not state.combat.metadata.get("combat_triggers")


def test_cables_modifies_only_automatic_first_orb_callbacks():
    state = _orb_state(
        orbs=("lightning", "lightning"), relics=("gold_plated_cables",), statuses={"loop": 1}
    )
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 88  # Automatic6+3 and manual Loop3.
    state = _orb_state(orbs=("plasma",), relics=("gold_plated_cables",), statuses={"loop": 1})
    assert _end_turn(_reload(state)).player.energy == 23  # Manual1 plus automatic2.


def test_metronome_activates_once_on_seventh_channel_and_keeps_counting():
    channel = _simple("channel", effects={"channel_orb": {"orb": "frost", "amount": 2}})
    state = _orb_state(channel, relics=("metronome",))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {**state.combat.metadata, "relic_counters": {"metronome": 6}},
                }
            )
        }
    )
    state = _play(_reload(state), "channel")
    assert state.combat.monsters[0].hp == 70
    assert state.combat.metadata["relic_counters"]["metronome"] == 8
    assert (
        len(
            [
                e
                for e in _events(state)
                if e.source_id == "metronome" and e.kind == "monster_damaged"
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "character,slots,channeled",
    [
        ("IRONCLAD", 1, True),
        ("SILENT", 1, True),
        ("REGENT", 1, True),
        ("NECROBINDER", 1, True),
        ("DEFECT", 0, False),
        ("TEST", 0, False),
    ],
)
def test_first_channel_gives_one_slot_only_to_characters_with_zero_base_slots(
    character,
    slots,
    channeled,
):
    channel = _simple("channel", effects={"channel_orb": {"orb": "frost"}})
    state = _orb_state(channel)
    state = state.model_copy(
        update={
            "character_id": character,
            "combat": state.combat.model_copy(update={"orb_slots": 0}),
        }
    )
    state = _play(_reload(state), "channel")
    assert state.combat.orb_slots == slots
    assert bool(state.combat.orbs) is channeled


@pytest.mark.parametrize(
    "block,expected_hp,expected_enemy_hp",
    [
        (0, 70, 94),
        (10, 80, 97),
    ],
)
def test_emotion_chip_uses_actual_prior_turn_hp_loss_and_resets_when_no_loss(
    block,
    expected_hp,
    expected_enemy_hp,
):
    state = _orb_state(orbs=("lightning",), relics=("emotion_chip",))
    player = state.player.model_copy(update={"block": block})
    monster = state.combat.monsters[0].model_copy(update={"intent": "Attack", "intent_damage": 10})
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(
                update={
                    "player": player,
                    "monsters": (monster,),
                }
            ),
        }
    )
    state = _end_turn(_reload(state))
    assert state.player.hp == expected_hp
    assert state.combat.monsters[0].hp == expected_enemy_hp
    assert state.combat.metadata["lost_hp_previous_turn"] is (block == 0)
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (
                        state.combat.monsters[0].model_copy(
                            update={"intent": "unknown", "intent_damage": 0}
                        ),
                    ),
                }
            )
        }
    )
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == expected_enemy_hp - 3
    assert state.combat.metadata["lost_hp_previous_turn"] is False


def test_spinner_channels_after_energy_reset_before_hand_draw():
    state = _orb_state(
        _source("SPINNER", upgraded=True),
        orbs=("plasma", "frost", "dark"),
        draw=(_simple("fresh"),),
        draw_per_turn=1,
    )
    state = _play(state, "spinner")
    assert [orb.orb_id for orb in state.combat.orbs] == ["frost", "dark", "glass"]
    assert state.player.energy == 21  # Cost1, overflow Plasma evoke2.
    state = _end_turn(_reload(state))
    assert [orb.orb_id for orb in state.combat.orbs] == ["dark", "glass", "glass"]
    assert [orb.value for orb in state.combat.orbs] == [12, 3, 4]
    events = _events(state)
    assert next(
        i for i, e in enumerate(events) if e.kind == "orb_channeled" and e.source_id == "spinner"
    ) < next(i for i, e in enumerate(events) if e.kind == "card_drawn")


def test_spinner_combat_ending_overflow_stops_later_turn_start_callbacks():
    state = _orb_state(
        orbs=("lightning",),
        relics=("happy_flower",),
        statuses={"spinner": 1},
        draw=(_simple("fresh"),),
        draw_per_turn=1,
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orb_slots": 1,
                    "monsters": (state.combat.monsters[0].model_copy(update={"hp": 10}),),
                }
            )
        }
    )
    state = _end_turn(_reload(state))
    assert state.phase == RunPhase.REWARD
    assert not state.combat.metadata.get("relic_counters", {}).get("happy_flower", 0)
    assert not any(e.kind == "card_drawn" for e in _events(state))
