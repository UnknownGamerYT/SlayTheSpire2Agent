"""Independent state-machine and side-end outcomes from stable source."""

from __future__ import annotations

import json
from random import Random

import pytest
from combat.test_stable_enemy_lifecycle import CACHE, _end, _fight

from sts2sim import load_state, serialize
from sts2sim.engine.transitions import _monster_has_spawn_capacity
from sts2sim.mechanics.monsters import (
    build_monster_definitions,
    next_monster_move,
    next_move_counts,
    resolve_monster_move,
)


class FixedRng(Random):
    def __init__(self, value):
        super().__init__(0)
        self.value = value
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.value


@pytest.mark.parametrize(
    "monster_id,start,expected",
    [
        ("LEAF_SLIME_S", "TACKLE", ["GOOP", "TACKLE", "GOOP"]),
        ("PHROG_PARASITE", "INFECT", ["LASH", "INFECT", "LASH"]),
        ("TWIG_SLIME_M", "STICKY_SHOT", ["POKEY_POUNCE", "POKEY_POUNCE", "STICKY_SHOT"]),
        ("SLITHERING_STRANGLER", "CONSTRICT", ["THWACK", "CONSTRICT", "THWACK"]),
        ("SCROLL_OF_BITING", "CHOMP", ["MORE_TEETH", "CHEW", "CHOMP"]),
        ("SPECTRAL_KNIGHT", "HEX", ["SOUL_SLASH", "SOUL_SLASH", "SOUL_FLAME"]),
    ],
)
def test_source_random_branch_rules_replace_uniform_move_fallback(monster_id, start, expected):
    definition = build_monster_definitions(json.loads(CACHE.read_text()))[monster_id]
    move_id = start
    counts = {}
    observed = []
    for _ in expected:
        counts = next_move_counts(counts, move_id)
        move_id = next_monster_move(definition, move_id, FixedRng(0.1), move_counts=counts).move_id
        observed.append(move_id)
    assert observed == expected


def test_flyconid_all_zero_source_cooldowns_still_draw_then_choose_first_branch():
    definition = build_monster_definitions(json.loads(CACHE.read_text()))["FLYCONID"]
    counts = {}
    for move in ("VULNERABLE_SPORES", "FRAIL_SPORES", "SMASH"):
        counts = next_move_counts(counts, move)
    rng = FixedRng(0.5)
    next_move = next_monster_move(definition, "SMASH", rng, move_counts=counts)
    assert next_move.move_id == "VULNERABLE_SPORES"
    assert rng.draws == 1  # Rng.NextFloat(0) still consumes one random value.


def test_fake_merchant_enrage_source_cooldown_counts_recent_moves():
    definition = build_monster_definitions(json.loads(CACHE.read_text()))["FAKE_MERCHANT_MONSTER"]
    counts = {}
    for move in ("ENRAGE", "SWIPE", "SPEW_COINS"):
        counts = next_move_counts(counts, move)
    before = resolve_monster_move(
        definition, "RAND_MOVE", FixedRng(0.99), previous_move_id="SPEW_COINS", move_counts=counts
    )
    assert before.move_id == "THROW_RELIC"
    counts = next_move_counts(counts, "THROW_RELIC")
    after = resolve_monster_move(
        definition, "RAND_MOVE", FixedRng(0.99), previous_move_id="THROW_RELIC", move_counts=counts
    )
    assert after.move_id == "ENRAGE"


def test_ritual_skips_application_side_and_retains_future_timing_after_reload():
    state = _end(_fight(("DAMP_CULTIST",)))
    assert state.combat.monsters[0].statuses["ritual"] == 5
    assert state.combat.monsters[0].statuses.get("strength", 0) == 0
    state = _end(load_state(serialize(state)))
    assert state.combat.monsters[0].statuses["strength"] == 5


def test_enemy_end_powers_resolve_after_all_enemy_attacks():
    state = _end(_fight(("ZAPBOT", "ZAPBOT")))
    events = state.combat.last_events
    attack_indices = [i for i, event in enumerate(events) if event.kind == "player_damaged"]
    gain_indices = [
        i for i, event in enumerate(events) if event.metadata.get("trigger") == "high_voltage"
    ]
    assert len(gain_indices) == 2
    assert max(attack_indices) < min(gain_indices)


def test_rat_two_turn_delay_ally_plans_and_global_backup_limit():
    state = _fight(("TWO_TAILED_RAT", "TWO_TAILED_RAT"))
    definition = build_monster_definitions(json.loads(CACHE.read_text()))["TWO_TAILED_RAT"]
    assert not _monster_has_spawn_capacity(state.combat, 0, definition)
    monsters = list(state.combat.monsters)
    monsters[0] = monsters[0].model_copy(
        update={
            "metadata": {
                **monsters[0].metadata,
                "rat_turns_until_summonable": 0,
            }
        }
    )
    combat = state.combat.model_copy(update={"monsters": tuple(monsters)})
    assert _monster_has_spawn_capacity(combat, 0, definition)
    monsters[1] = monsters[1].model_copy(update={"move_id": "CALL_FOR_BACKUP"})
    assert not _monster_has_spawn_capacity(
        combat.model_copy(update={"monsters": tuple(monsters)}), 0, definition
    )

    monsters[1] = monsters[1].model_copy(update={"move_id": "SCRATCH"})
    monsters[0] = monsters[0].model_copy(
        update={
            "metadata": {
                **monsters[0].metadata,
                "rat_backup_count": 3,
            }
        }
    )
    assert not _monster_has_spawn_capacity(
        combat.model_copy(update={"monsters": tuple(monsters)}), 0, definition
    )


def test_guardbot_only_blocks_fabricators_and_cubex_starts_with_thirteen_block():
    state = _fight(("GUARDBOT", "FABRICATOR", "STABBOT"))
    state = _end(state)
    assert state.combat.monsters[0].block == 0
    assert state.combat.monsters[1].block == 15
    assert state.combat.monsters[2].block == 0
    assert _fight(("CUBEX_CONSTRUCT",)).combat.monsters[0].block == 13


def test_spiky_enemies_remove_thorns_on_their_attack_move():
    state = _fight(("TOADPOLE",))
    monster = state.combat.monsters[0].model_copy(
        update={
            "move_id": "SPIKE_SPIT",
            "statuses": {"thorns": 2},
        }
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster,),
                }
            )
        }
    )
    state = _end(state)
    assert "thorns" not in state.combat.monsters[0].statuses
    state = _end(_fight(("SPINY_TOAD",)))
    assert state.combat.monsters[0].statuses["thorns"] == 5
    state = _end(state)
    assert "thorns" not in state.combat.monsters[0].statuses


def test_living_fog_bloat_summons_one_bomb_before_its_attack_each_use():
    state = _fight(("LIVING_FOG",))
    for _ in range(3):
        monsters = list(state.combat.monsters)
        monsters[0] = monsters[0].model_copy(update={"move_id": "BLOAT"})
        state = state.model_copy(
            update={
                "combat": state.combat.model_copy(
                    update={
                        "monsters": tuple(monsters),
                    }
                )
            }
        )
        state = _end(state)
        events = state.combat.last_events
        summoned = [i for i, event in enumerate(events) if event.kind == "monster_summoned"]
        fog_hits = [
            i
            for i, event in enumerate(events)
            if event.kind == "player_damaged" and event.source_id == "LIVING_FOG"
        ]
        assert len(summoned) == 1
        assert summoned[0] < fog_hits[0]
