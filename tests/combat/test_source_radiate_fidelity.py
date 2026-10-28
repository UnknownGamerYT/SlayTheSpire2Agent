"""Radiate source snapshot regressions through the public step API."""

import json
from pathlib import Path

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, serialize, step
from sts2sim.engine import CardEnchantment
from sts2sim.engine.transitions import _card_from_spec

_ROWS = json.loads(
    (Path(__file__).parents[1] / "fixtures/source_radiate_fidelity.json").read_text()
)


def _source(card_id, *, upgraded=False, instance_id=None):
    row = next(row for row in _ROWS if row["id"] == card_id)
    return _card_from_spec(
        {**row, "upgraded": upgraded, "instance_id": instance_id or card_id.lower()}, 0
    )


def _hits(state):
    return [
        event
        for event in state.combat.last_events
        if event.kind == "monster_damaged" and event.source_id == "radiate"
    ]


@pytest.mark.parametrize("upgraded,damage", [(False, 3), (True, 4)])
@pytest.mark.parametrize("gained", [0, 1, 2])
def test_radiate_has_one_aoe_hit_per_star_actually_gained(upgraded, damage, gained):
    gain = _source("GATHER_LIGHT" if gained == 1 else "VENERATE")
    state = _state(_source("RADIATE", upgraded=upgraded), gain, statuses={"strength": 2})
    monster = state.combat.monsters[0]
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster, monster.model_copy(update={"monster_id": "other"})),
                }
            )
        }
    )
    if gained:
        state = _play(state, gain.instance_id)
    state = _play(_reload(state), "radiate")
    assert [monster.hp for monster in state.combat.monsters] == [100 - (damage + 2) * gained] * 2
    assert len(_hits(state)) == gained * 2
    assert state.combat.metadata.get("stars_gained_this_turn", 0) == gained
    assert state.combat.metadata["attacks_played_this_turn"] == 1


def test_radiate_per_hit_vulnerable_rounding_matches_wiki_notes():
    # https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Radiate#Notes
    state = _state(_source("RADIATE"), _source("VENERATE"))
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"vulnerable": 2}})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(_play(state, "venerate"), "radiate")
    assert state.combat.monsters[0].hp == 92
    assert [event.amount for event in _hits(state)] == [4, 4]


def test_radiate_does_not_count_spent_stars_or_initial_resources():
    state = _state(_source("STARDUST"), _source("RADIATE"))
    player = state.player.model_copy(update={"resources": {"star": 2}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _play(state, "stardust")
    before = state.combat.monsters[0].hp
    state = _play(_reload(state), "radiate")
    assert state.combat.monsters[0].hp == before
    assert not _hits(state)
    assert state.combat.metadata.get("stars_gained_this_turn", 0) == 0


def test_radiate_gain_counter_resets_each_turn_and_preserves_future_gains():
    radiate = _source("RADIATE")
    state = _play(_state(radiate, _source("HIDDEN_CACHE")), "hidden_cache")
    assert state.combat.metadata["stars_gained_this_turn"] == 1
    state = _end_turn(_reload(state))
    assert state.combat.metadata["stars_gained_this_turn"] == 3
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"hand": (radiate,)})}
    )
    state = _play(_reload(state), "radiate")
    assert len(_hits(state)) == 3
    state = _end_turn(_reload(state))
    assert state.combat.metadata.get("stars_gained_this_turn", 0) == 0


def test_radiate_counts_registered_power_gains_once_before_its_own_play_trigger():
    state = _state(_source("THE_SEALED_THRONE"), _source("VENERATE"), _source("RADIATE"))
    state = _play(_play(state, "the_sealed_throne"), "venerate")
    assert state.combat.metadata["stars_gained_this_turn"] == 3
    state = _play(_reload(state), "radiate")
    assert len(_hits(state)) == 3
    assert state.combat.monsters[0].hp == 91
    assert state.combat.metadata["stars_gained_this_turn"] == 4


def test_radiate_counts_kill_resource_gains():
    state = _state(_source("KNOCKOUT_BLOW"), _source("RADIATE"))
    monster = state.combat.monsters[0]
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (
                        monster.model_copy(update={"hp": 20}),
                        monster.model_copy(update={"monster_id": "other"}),
                    ),
                }
            )
        }
    )
    state = _play(state, "knockout_blow")
    assert state.combat.metadata["stars_gained_this_turn"] == 5
    state = _play(_reload(state), "radiate")
    assert len(_hits(state)) == 5
    assert state.combat.monsters[1].hp == 85


def test_radiate_counts_potion_star_gains():
    state = _state(_source("RADIATE"))
    state = state.model_copy(update={"potions": ("star_potion",)})
    state = step(state, next(a for a in legal_actions(state) if a.type == "use_potion"))
    gained = state.player.resources["star"] - 20
    assert gained > 0
    assert state.combat.metadata["stars_gained_this_turn"] == gained
    state = _play(_reload(state), "radiate")
    assert len(_hits(state)) == gained


def test_radiate_replays_use_same_counter_and_match_after_reload():
    radiate = _source("RADIATE").model_copy(
        update={"enchantments": (CardEnchantment(keyword="glam", amount=1),)}
    )
    state = _play(_state(radiate, _source("VENERATE")), "venerate")
    played = _play(state, "radiate")
    assert serialize(played) == serialize(_play(_reload(state), "radiate"))
    assert len(_hits(played)) == 4
    assert played.combat.monsters[0].hp == 88
    assert played.combat.metadata["stars_gained_this_turn"] == 2


def test_radiate_counts_divine_right_combat_start_stars_on_first_turn():
    from combat.test_relic_engine_integration import _enter_combat

    row = next(row for row in _ROWS if row["id"] == "RADIATE")
    state = _enter_combat((row,), relics=("divine_right",), source_data_extra={"cards": _ROWS})
    assert state.player.resources["star"] == 3
    assert state.combat.metadata["stars_gained_this_turn"] == 3
    card = next(card for card in state.combat.hand if card.card_id == "radiate")
    before = state.combat.monsters[0].hp
    state = _play(_reload(state), card.instance_id)
    assert state.combat.monsters[0].hp == before - 9


def test_radiate_counter_ignores_zero_gain_and_clamped_resource_loss():
    zero = _simple("zero", effects={"player_resource": {"resource": "star", "amount": 0}})
    loss = _simple("loss", effects={"player_resource": {"resource": "star", "amount": -999}})
    state = _state(_source("RADIATE"), zero, loss)
    state = _play(_play(state, "zero"), "loss")
    assert state.player.resources["star"] == 0
    state = _play(_reload(state), "radiate")
    assert not _hits(state)
    assert state.combat.metadata.get("stars_gained_this_turn", 0) == 0


def test_radiate_counts_genesis_turn_start_power_gain_after_reset():
    radiate = _source("RADIATE")
    state = _play(_state(_source("GENESIS"), radiate), "genesis")
    assert state.combat.metadata.get("stars_gained_this_turn", 0) == 0
    state = _end_turn(_reload(state))
    assert state.combat.metadata["stars_gained_this_turn"] == 2
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"hand": (radiate,)})}
    )
    state = _play(_reload(state), "radiate")
    assert len(_hits(state)) == 2
