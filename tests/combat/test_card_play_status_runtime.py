from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_glam import _combat, _end_turn, _play

from sts2sim import legal_actions, load_state, serialize
from sts2sim.engine import CardEnchantment, CardInstance
from sts2sim.engine.transitions import _card_from_spec


def _source(card_id, *, upgraded=False):
    path = Path("data/cache/eng/cards.json")
    if not path.exists():
        pytest.skip("Stable source cache not present")
    rows = json.loads(path.read_text())
    row = next(row for row in rows if row["id"] == card_id)
    return _card_from_spec({**row, "upgraded": upgraded, "instance_id": card_id.lower()}, 0)


def _simple(card_id, *, effects=None, cost=0, card_type="skill", target="self", **kw):
    return CardInstance(
        instance_id=card_id,
        card_id=card_id,
        cost=cost,
        type=card_type,
        target=target,
        effects=effects or {},
        **kw,
    )


def _state(*cards, statuses=None, draw=(), relics=(), draw_per_turn=0):
    state = _combat(cards[0], relics=relics)
    player = state.player.model_copy(
        update={
            "energy": 20,
            "max_energy": 20,
            "statuses": statuses or {},
            "resources": {"star": 20},
        }
    )
    return state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(
                update={
                    "player": player,
                    "hand": tuple(cards),
                    "draw_pile": tuple(draw),
                    "draw_per_turn": draw_per_turn,
                }
            ),
        }
    )


def _reload(state):
    return load_state(json.loads(json.dumps(serialize(state))))


def test_source_afterimage_has_one_consumer_and_stacks_once_per_play() -> None:
    power = _source("AFTERIMAGE")
    skill = _simple("block", effects={"block": 2})
    state = _play(_state(power, skill), "afterimage")
    assert state.player.block == 1
    state = _play(_reload(state), "block")
    assert state.player.block == 4
    assert not state.combat.metadata.get("combat_triggers")


def test_source_arsenal_counts_each_created_physical_card_once() -> None:
    power = _source("ARSENAL")
    generated = {"card_id": "token", "name": "Token", "type": "skill", "cost": 0}
    generator = _simple(
        "generator",
        effects={
            "sequence": [
                {"add_card_to_hand": {"card": generated}},
                {"add_card_to_hand": {"card": generated}},
            ]
        },
    )
    state = _play(_play(_state(power, generator), "arsenal"), "generator")
    assert state.player.statuses["strength"] == 2
    tokens = [card for card in state.combat.hand if card.card_id == "token"]
    assert len(tokens) == len({card.instance_id for card in tokens}) == 2


def test_source_feral_overrides_exhaust_only_for_first_zero_cost_attack() -> None:
    power = _source("FERAL")
    attack = _simple(
        "attack", effects={"damage": 6}, card_type="attack", target="enemy", exhausts=True
    )
    state = _play(_state(power, attack, relics=("charons_ashes",)), "feral")
    state = _play(_reload(state), "attack")
    assert [card.instance_id for card in state.combat.hand] == ["attack"]
    assert not state.combat.exhaust_pile
    assert state.combat.monsters[0].hp == 94
    state = _play(state, "attack")
    assert not state.combat.hand
    assert [card.instance_id for card in state.combat.exhaust_pile] == ["attack"]
    assert state.combat.monsters[0].hp == 85


def test_feral_counts_zero_attacks_played_before_power_and_resets_next_turn() -> None:
    first = _simple("first", effects={"damage": 1}, card_type="attack", target="enemy")
    second = first.model_copy(update={"instance_id": "second", "card_id": "second"})
    power = _source("FERAL")
    state = _play(_play(_play(_state(first, power, second), "first"), "feral"), "second")
    assert not state.combat.hand
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (first,),
                    "discard_pile": (power, second),
                    "draw_per_turn": 1,
                }
            )
        }
    )
    state = _play(_end_turn(_reload(state)), "first")
    assert [card.instance_id for card in state.combat.hand] == ["first"]


def test_lethality_applies_to_first_attack_only_and_not_its_replay() -> None:
    power = _source("LETHALITY")
    attack = _simple(
        "attack",
        effects={"damage": 6},
        card_type="attack",
        target="enemy",
        enchantments=(CardEnchantment(keyword="glam", amount=1),),
    )
    state = _play(_play(_state(power, attack), "lethality"), "attack")
    assert state.combat.monsters[0].hp == 85  # Nine, then six.
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (attack,),
                    "discard_pile": (power,),
                    "draw_per_turn": 1,
                }
            )
        }
    )
    state = _play(_end_turn(_reload(state)), "attack")
    assert state.combat.monsters[0].hp == 76


def test_lethality_does_not_rearm_if_gained_after_first_attack() -> None:
    attack = _simple("attack", effects={"damage": 6}, card_type="attack", target="enemy")
    second = attack.model_copy(update={"instance_id": "second", "card_id": "second"})
    state = _state(attack, _source("LETHALITY"), second)
    for instance_id in ("attack", "lethality", "second"):
        state = _play(state, instance_id)
    assert state.combat.monsters[0].hp == 88


@pytest.mark.parametrize("upgraded,bonus", [(False, 9), (True, 12)])
def test_phantom_blades_retains_shivs_and_boosts_only_first_each_turn(upgraded, bonus) -> None:
    power = _source("PHANTOM_BLADES", upgraded=upgraded)
    shiv = _simple("shiv", effects={"damage": 4}, card_type="attack", target="enemy", exhausts=True)
    other = shiv.model_copy(update={"instance_id": "other"})
    state = _play(_state(power, shiv, other), "phantom_blades")
    assert len(state.combat.hand) == 2  # The power does not create a Shiv.
    state = _play(_reload(state), "shiv")
    assert state.combat.monsters[0].hp == 100 - 4 - bonus
    state = _end_turn(state)
    assert [card.instance_id for card in state.combat.hand] == ["other"]
    state = _play(state, "other")
    assert state.combat.monsters[0].hp == 100 - 2 * (4 + bonus)


def test_iteration_bonus_draw_does_not_recurse_on_another_status_and_rearms() -> None:
    power = _source("ITERATION")
    draw = _simple("draw", effects={"draw": 1})
    cards = tuple(_simple(f"status:{i}", card_type="status") for i in range(4))
    state = _play(_play(_state(power, draw, draw=cards), "iteration"), "draw")
    assert len(state.combat.hand) == 3
    assert [card.instance_id for card in state.combat.draw_pile] == ["status:3"]
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 1})}
    )
    state = _end_turn(_reload(state))
    assert state.combat.metadata["card_status_turn_counters"]["iteration"] == 1
    assert len(state.combat.hand) == 3


def test_speedster_hits_once_per_card_drawn_but_not_start_turn_draws() -> None:
    power = _source("SPEEDSTER")
    draw = _simple("draw", effects={"draw": 2})
    cards = tuple(_simple(f"card:{i}") for i in range(4))
    state = _play(_play(_state(power, draw, draw=cards, draw_per_turn=2), "speedster"), "draw")
    assert state.combat.monsters[0].hp == 96
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 96
    assert len(state.combat.hand) == 2


@pytest.mark.parametrize("upgraded,damage", [(False, 4), (True, 6)])
def test_serpent_form_triggers_for_power_itself_and_following_card(upgraded, damage) -> None:
    power = _source("SERPENT_FORM", upgraded=upgraded)
    skill = _simple("skill")
    state = _play(_play(_state(power, skill), "serpent_form"), "skill")
    assert state.combat.monsters[0].hp == 100 - damage * 2


def test_signal_boost_consumes_one_charge_for_each_of_next_two_powers() -> None:
    boost = _source("SIGNAL_BOOST")
    second_boost = boost.model_copy(update={"instance_id": "boost:2"})
    first = _simple(
        "first", card_type="power", effects={"apply_status": {"target": "self", "strength": 1}}
    )
    second = first.model_copy(update={"instance_id": "second", "card_id": "second"})
    state = _state(boost, second_boost, first, second)
    for instance_id in ("signal_boost", "boost:2", "first"):
        state = _play(state, instance_id)
    assert state.player.statuses["strength"] == 2
    assert state.player.statuses["signalboost"] == 1
    state = _play(_reload(state), "second")
    assert state.player.statuses["strength"] == 4
    assert not state.player.statuses.get("signalboost")
    assert len(state.combat.cards_played_this_turn) == 6


def test_strangle_is_hp_loss_after_card_effects_and_expires_next_turn() -> None:
    power = _source("STRANGLE")
    skill = _simple("skill")
    other = _simple("other")
    state = _state(power, skill, other)
    monster = state.combat.monsters[0].model_copy(
        update={"block": 100, "statuses": {"vulnerable": 1}}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state, "strangle")
    assert state.combat.monsters[0].hp == 100
    assert state.combat.monsters[0].block == 88  # Only the printed eight damage uses Vulnerable.
    state = _play(_reload(state), "skill")
    assert state.combat.monsters[0].hp == 98
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (other,),
                    "hand": (),
                    "draw_per_turn": 1,
                }
            )
        }
    )
    state = _play(_end_turn(state), "other")
    assert state.combat.monsters[0].hp == 98
    assert not state.combat.monsters[0].statuses.get("strangle")


def test_artifact_blocks_strangle_and_its_future_hp_loss() -> None:
    power = _source("STRANGLE")
    state = _state(power)
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"artifact": 1}})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state, "strangle")
    assert state.combat.monsters[0].hp == 92
    assert not state.combat.monsters[0].statuses.get("strangle")
    assert not state.combat.monsters[0].statuses.get("artifact")


def test_juggernaut_triggers_once_per_block_instance_including_afterimage() -> None:
    power = _source("JUGGERNAUT")
    block = _simple("block", effects={"block": 100})
    state = _play(
        _state(power, block, statuses={"afterimage": 1, "strength": 100, "weak": 1}), "juggernaut"
    )
    assert state.combat.monsters[0].hp == 94
    state = _play(_reload(state), "block")
    assert state.combat.monsters[0].hp == 82  # Two instances, regardless of Block/Strength.


def test_juggernaut_handles_metallicize_before_enemy_turn() -> None:
    state = _state(_simple("unused"), statuses={"juggernaut": 6, "metallicize": 3})
    state = _end_turn(state)
    assert state.combat.monsters[0].hp == 94


def test_void_form_ends_after_replays_and_frees_two_manual_cards_energy_and_stars() -> None:
    power = _source("VOID_FORM").model_copy(
        update={"enchantments": (CardEnchantment(keyword="glam", amount=1),)}
    )
    cards = tuple(
        _simple(f"card:{i}", cost=9, custom={"star_cost": 9}, effects={"block": 1})
        for i in range(5)
    )
    state = _play(_state(power, draw=cards, draw_per_turn=5), "void_form")
    assert state.combat.turn == 2
    assert state.player.statuses["voidform"] == 4
    player = state.player.model_copy(update={"energy": 0, "resources": {"star": 0}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    for index in range(4):
        state = _play(_reload(state), f"card:{index}")
        assert state.player.energy == state.player.resources["star"] == 0
    assert not any(action.type == "play_card" for action in legal_actions(state))


def test_void_form_quota_excludes_replays_and_autoplays_but_x_cost_still_spends() -> None:
    parent = _simple(
        "parent",
        cost=9,
        effects={"play_top_card": 1},
        enchantments=(CardEnchantment(keyword="glam", amount=1),),
    )
    child = _simple("child", cost=9)
    second = _simple("second", cost=9, custom={"star_cost": 9})
    state = _play(_state(parent, second, draw=(child,), statuses={"voidform": 2}), "parent")
    assert state.combat.metadata["card_status_turn_counters"]["manual_cards"] == 1
    state = _play(_reload(state), "second")
    assert state.player.energy == 20
    assert state.player.resources["star"] == 20
    xcard = _simple(
        "x",
        cost=-1,
        effects={"block": {"per_energy": 1}},
        custom={"star_cost": "X", "x_cost": True},
    )
    state = _play(_state(xcard, statuses={"voidform": 2}), "x")
    assert state.player.energy == state.player.resources["star"] == 0
    assert state.player.block == 20


def test_generated_overflow_goes_to_discard_and_triggers_arsenal_once_per_card() -> None:
    token = {"instance_id": "same-explicit-id", "card_id": "token", "type": "skill"}
    generator = _simple("generator", effects={"add_card_to_hand": {"card": token, "count": 3}})
    junk = tuple(_simple(f"junk:{index}") for index in range(9))
    state = _play(_state(generator, *junk, statuses={"arsenal": 2}), "generator")
    assert len(state.combat.hand) == 10
    tokens = [
        card for card in state.combat.hand + state.combat.discard_pile if card.card_id == "token"
    ]
    assert len(tokens) == len({card.instance_id for card in tokens}) == 3
    assert len([card for card in state.combat.discard_pile if card.card_id == "token"]) == 2
    assert state.player.statuses["strength"] == 6
    created = [
        event
        for event in state.replay_log[-1].events
        if event.target_id in {card.instance_id for card in tokens}
    ]
    assert sum(event.kind == "card_added_to_hand" for event in created) == 1
    assert sum(event.kind == "card_added_to_discard" for event in created) == 2


def test_nested_autoplay_creation_does_not_trigger_arsenal_twice() -> None:
    parent = _simple("parent", effects={"play_top_card": 1})
    child = _simple("child", effects={"add_card_to_hand": {"card": {"card_id": "token"}}})
    state = _play(_state(parent, draw=(child,), statuses={"arsenal": 1}), "parent")
    assert state.player.statuses["strength"] == 1
    assert len([card for card in state.combat.hand if card.card_id == "token"]) == 1


def test_random_free_creation_overflow_preserves_count_and_unique_ids() -> None:
    generator = _simple("generator", effects={"add_random_card_to_hand": 3})
    state = _state(
        generator, *(_simple(f"junk:{index}") for index in range(9)), statuses={"arsenal": 1}
    )
    state = _play(state, "generator")
    assert len(state.combat.hand) == 10
    assert len(state.combat.discard_pile) == 3
    assert len({card.instance_id for card in state.combat.hand + state.combat.discard_pile}) == 13
    assert state.player.statuses["strength"] == 3
    events = state.replay_log[-1].events
    assert sum(event.kind == "card_added_to_hand" for event in events) == 1
    assert sum(event.kind == "card_added_to_discard" for event in events) == 2


def test_random_source_pool_creation_overflow_preserves_count_and_unique_ids() -> None:
    generator = _simple("generator", effects={"add_random_card_to_hand": {"count": 3}})
    state = _state(
        generator, *(_simple(f"junk:{index}") for index in range(9)), statuses={"arsenal": 1}
    )
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "card_pool": (
                    {
                        "id": "TOKEN",
                        "name": "Token",
                        "type": "Skill",
                        "color": "test",
                        "cost": 0,
                    },
                ),
            }
        }
    )
    state = _play(state, "generator")
    assert len(state.combat.hand) == 10
    tokens = [
        card for card in state.combat.hand + state.combat.discard_pile if card.card_id == "token"
    ]
    assert len(tokens) == len({card.instance_id for card in tokens}) == 3
    assert state.player.statuses["strength"] == 3


def test_juggernaut_relic_block_triggers_immediately_and_once() -> None:
    discard = _simple("discard", effects={"discard_choice": 1})
    junk = _simple("junk")
    state = _play(
        _state(discard, junk, statuses={"juggernaut": 6}, relics=("tough_bandages",)), "discard"
    )
    from sts2sim import step

    action = next(action for action in legal_actions(state) if action.type == "discard_card")
    state = step(_reload(state), action)
    assert state.player.block == 3
    assert state.combat.monsters[0].hp == 94
