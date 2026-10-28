"""Source event encounters, extra rewards and Battleworn Dummy.Resume outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.content.source_potion_pools import source_potion_options
from sts2sim.engine import EventOptionState, EventState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.engine.rng import RunRng, capture_random_state, random_from_state, random_stream
from sts2sim.engine.transitions import _start_event_option_combat
from sts2sim.mechanics.relic_bag_runtime import source_relic_rarity

CACHE = Path(__file__).resolve().parents[2] / "data/cache/eng"


def _fight(event_id, *, setting=1, relics=(), deck=None, metadata=None, source_rng=False):
    state = new_run(
        seed=222,
        character_id="TEST",
        ascension=0,
        source_data={
            "monsters": json.loads((CACHE / "monsters.json").read_text()),
            "encounters": json.loads((CACHE / "encounters.json").read_text()),
            "cards": json.loads((CACHE / "cards.json").read_text()),
            "deck": deck
            or [
                {
                    "id": "kill",
                    "type": "attack",
                    "target": "all_enemies",
                    "cost": 0,
                    "effects": {"all_damage": 9999},
                    "rarity": "rare",
                }
            ],
            "player": {"hp": 50, "max_hp": 80},
            "relics": relics,
        },
    )
    node = MapNodeState(node_id="event", act=1, floor=1, lane=0, kind=RoomKind.EVENT)
    option = EventOptionState(
        option_id=f"setting_{setting}" if event_id == "battleworn_dummy" else "fight",
        title="Fight",
        description="Fight.",
    )
    event = EventState(
        event_id=event_id,
        name=event_id,
        options=(option,),
        resolved_option_id=option.option_id,
        metadata=metadata or {},
    )
    state = state.model_copy(
        update={
            "player": state.player.model_copy(update={"hp": 50}),
            "phase": RunPhase.EVENT,
            "event": event,
            "map": MapState(nodes=(node,), current_node_id="event"),
        }
    )
    if source_rng:
        state = state.model_copy(update={"rng": capture_random_state(RunRng(222))})
    return _start_event_option_combat(state, event, option, ())[0]


def _kill(state):
    return step(
        state,
        next(
            a
            for a in legal_actions(state)
            if a.type == "play_card"
            and a.card_instance_id
            == next(c.instance_id for c in state.combat.hand if c.card_id == "kill")
        ),
    )


def _end(state):
    return step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))


@pytest.mark.parametrize(
    "event_id,roster",
    [
        ("dense_vegetation", ["WRIGGLER"] * 4),
        ("punch_off", ["PUNCH_CONSTRUCT"] * 2),
        ("the_lantern_key", ["MYSTERIOUS_KNIGHT"]),
    ],
)
def test_shared_event_fights_select_the_source_encounter(event_id, roster):
    state = _fight(event_id)
    assert [m.metadata["source_monster_id"] for m in state.combat.monsters] == roster
    state = _kill(state)
    assert state.phase == RunPhase.REWARD
    assert 10 <= state.reward.gold <= 20
    assert len(state.reward.card_options) == 3
    if event_id == "punch_off":
        assert len(state.reward.relic_ids) == 1
        assert len(state.reward.potion_ids) == 1
    elif event_id == "the_lantern_key":
        assert state.reward.card_ids == ("lantern_key",)
    else:
        assert state.reward.relic_ids == ()
        assert state.reward.card_ids == ()


@pytest.mark.parametrize(
    "setting,monster", [(1, "BATTLE_FRIEND_V1"), (2, "BATTLE_FRIEND_V2"), (3, "BATTLE_FRIEND_V3")]
)
def test_dummy_setting_selects_only_its_model(setting, monster):
    state = _fight("battleworn_dummy", setting=setting)
    assert [m.metadata["source_monster_id"] for m in state.combat.monsters] == [monster]


def test_dummy_setting_one_uniform_potion_uses_one_rewards_draw_and_no_base_rewards():
    state = _fight("battleworn_dummy", source_rng=True)
    rng = random_from_state(state.rng)
    expected = random_stream(rng, "rewards").choice(source_potion_options("TEST"))
    expected_rng = capture_random_state(rng)
    state = _kill(state)
    assert state.reward.potion_id == expected
    assert state.reward.gold == 0
    assert state.reward.card_options == ()
    assert state.reward.relic_ids == ()
    assert state.rng == expected_rng


def test_dummy_setting_two_directly_upgrades_two_eligible_cards_including_eternal():
    deck = [
        {
            "id": "kill",
            "type": "attack",
            "target": "all_enemies",
            "cost": 0,
            "effects": {"all_damage": 9999},
        },
        {
            "id": "eligible",
            "type": "skill",
            "cost": 0,
            "tags": ("eternal",),
            "effects": {"block": 1},
        },
        {"id": "wound", "type": "status"},
        {"id": "burn", "type": "status"},
    ]
    state = _fight("battleworn_dummy", setting=2, deck=deck, source_rng=True)
    restored = load_state(serialize(state))
    state = _kill(state)
    assert serialize(state) == serialize(_kill(restored))
    assert state.phase == RunPhase.MAP
    assert state.reward is None
    assert {c.card_id: c.upgraded for c in state.master_deck} == {
        "kill": True,
        "eligible": True,
        "wound": False,
        "burn": False,
    }
    assert "event_fight_id" not in state.flags
    assert state.room_history == ("event",)


def test_dummy_setting_three_obtains_front_bag_relic_directly():
    state = _fight("battleworn_dummy", setting=3, source_rng=True)
    bag = {"Common": ("ANCHOR",), "Uncommon": ("PEN_NIB",), "Rare": ("POCKETWATCH",)}
    state = state.model_copy(
        update={
            "character_id": "IRONCLAD",
            "flags": {**state.flags, "player_relic_bag": bag},
        }
    )
    rng = random_from_state(state.rng)
    rarity = source_relic_rarity(random_stream(rng, "rewards")).title()
    state = _kill(state)
    assert state.phase == RunPhase.MAP
    assert state.reward is None
    assert bag[rarity][0].lower() in state.relics
    assert state.flags["player_relic_bag"][rarity] == ()


def test_dummy_timeout_still_runs_victory_hooks_but_awards_no_setting_reward():
    state = _fight("battleworn_dummy", relics=("burning_blood",), source_rng=True)
    before_rewards = state.rng.streams.get("rewards")
    state = _end(_end(_end(state)))
    assert state.phase == RunPhase.MAP
    assert state.reward is None
    assert state.player.hp == 56
    assert state.rng.streams.get("rewards") == before_rewards


def test_fake_merchant_reward_has_fixed_300_gold_plus_rewards_hook_and_unsold_only():
    state = _fight(
        "fake_merchant",
        relics=("amethyst_aubergine",),
        metadata={"fake_merchant_unsold_relic_ids": ("fake_anchor", "fake_mango")},
    )
    state = _kill(state)
    assert state.reward.gold == 315
    assert len(state.reward.card_options) == 3
    assert state.reward.relic_ids == ("fake_merchants_rug", "fake_anchor", "fake_mango")


def test_sold_out_fake_merchant_awards_only_rug_as_relic_extra():
    state = _fight("fake_merchant", metadata={"fake_merchant_unsold_relic_ids": ()})
    state = _kill(state)
    assert state.reward.gold == 300
    assert state.reward.relic_ids == ("fake_merchants_rug",)


def test_source_dummy_relic_resume_requires_authoritative_bag():
    from sts2sim.engine import EngineError

    state = _fight("battleworn_dummy", setting=3, source_rng=True)
    state = state.model_copy(update={"character_id": "IRONCLAD"})
    with pytest.raises(EngineError, match="source player relic bag"):
        _kill(state)


def test_source_fake_combat_requires_actual_inventory_instead_of_assuming_all_fakes():
    from sts2sim.engine import EngineError

    with pytest.raises(EngineError, match="source unsold inventory"):
        _fight("fake_merchant", source_rng=True)


def test_dummy_potion_resume_respects_actual_revealed_epoch_profile():
    state = _fight("battleworn_dummy", source_rng=True)
    state = state.model_copy(update={
        "character_id": "SILENT", "flags": {**state.flags, "revealed_epochs": []},
    })
    rng = random_from_state(state.rng)
    expected = random_stream(rng, "rewards").choice(
        source_potion_options("SILENT", unlocked_epochs=[]),
    )
    state = _kill(state)
    assert state.reward.potion_id == expected
    assert state.rng == capture_random_state(rng)


def test_dummy_relic_resume_preserves_stackable_circlet_fallback():
    state = _fight("battleworn_dummy", setting=3, source_rng=True)
    state = state.model_copy(update={
        "character_id": "IRONCLAD", "relics": ("circlet",),
        "flags": {**state.flags, "player_relic_bag": {
            "Common": (), "Uncommon": (), "Rare": (),
        }},
    })
    state = _kill(state)
    assert state.phase == RunPhase.MAP
    assert state.relics.count("circlet") == 2


def test_battleworn_timeout_retains_serialized_resume_and_victory_outcomes():
    state = _fight("battleworn_dummy", relics=("burning_blood",), source_rng=True)
    state = _end(_end(state))
    restored = load_state(serialize(state))
    assert serialize(_end(state)) == serialize(_end(restored))


def test_dummy_timeout_ticks_wongo_without_consuming_its_ready_reward():
    state = _fight("battleworn_dummy", relics=("wongos_mystery_ticket",), source_rng=True)
    from sts2sim.engine import transitions as t

    state, _ = t._apply_relic_pickup_effects(state, "wongos_mystery_ticket")
    flags = dict(state.flags)
    queue = list(t._delayed_event_rewards_from_flags(flags))
    queue[0] = {**queue[0], "remaining_combats": 1}
    state = state.model_copy(update={"flags": t._write_delayed_event_rewards(flags, queue)})
    state = _end(_end(_end(state)))
    assert state.phase == RunPhase.MAP
    assert state.reward is None
    assert t._delayed_event_rewards_from_flags(state.flags)[0]["remaining_combats"] == 0
    assert not state.flags.get("wongos_mystery_ticket_used_up", False)
    assert not any(e.kind == "delayed_event_reward_ready" for e in state.replay_log[-1].events)
