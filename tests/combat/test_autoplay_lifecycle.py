from __future__ import annotations

import json

from combat.test_glam import _card, _combat, _play
from combat.test_replay_hooks import _player_statuses

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardEnchantment, CardInstance, CardType, RunPhase


def _parent(effects):
    return _card(effects=effects, target="self", cost=0).model_copy(
        update={"type": CardType.SKILL, "enchantments": (), "custom": {}}
    )


def _with_cards(state, *, hand=None, draw=()):
    return state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": state.combat.hand if hand is None else hand,
                    "draw_pile": draw,
                }
            )
        }
    )


def _choose(state, action_type, instance_id):
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == action_type and action.card_instance_id == instance_id
        ),
    )


def test_top_autoplay_runs_card_power_and_monster_hooks_without_paying_costs() -> None:
    parent = _parent({"play_top_card": 1})
    child = CardInstance(
        instance_id="child",
        card_id="child",
        type="skill",
        target="self",
        cost=9,
        custom={"star_cost": 9},
        effects={"block": 1},
    )
    trigger = {"trigger": "card_played", "source_card_id": "listener", "effects": ({"block": 4},)}
    state = _with_cards(_combat(parent, metadata={"combat_triggers": (trigger,)}), draw=(child,))
    state = _player_statuses(state, afterimage=2)
    monster = state.combat.monsters[0].model_copy(
        update={"statuses": {"enrage": 3, "vital_spark": 1, "withering_presence": 2}}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state)
    assert state.player.block == 13  # Child block + two Afterimages + two power triggers.
    assert state.player.energy == 3
    assert state.player.resources["star"] == 3
    assert state.player.statuses["tainted"] == 2
    assert state.combat.monsters[0].statuses["strength"] == 6
    assert state.combat.metadata["cards_played_this_combat"] == 2
    assert state.combat.metadata["skills_played_this_turn"] == 2
    assert len([card for card in state.combat.hand if card.card_id == "wither"]) == 1
    assert sorted(card.instance_id for card in state.combat.discard_pile) == ["child", "glam:1"]


def test_parent_and_autoplayed_child_both_generate_potions() -> None:
    parent = _parent({"play_top_card": 1, "add_random_potion": 1})
    child = CardInstance(
        instance_id="child",
        card_id="child",
        type="skill",
        target="self",
        effects={"add_random_potion": 1},
    )
    state = _with_cards(_combat(parent), draw=(child,))
    state = state.model_copy(update={"flags": {**state.flags, "potion_pool": ("fire_potion",)}})
    state = _play(state)
    assert state.potions == ("fire_potion", "fire_potion")
    assert "replay_potions" not in state.combat.metadata


def test_fresh_autoplays_respect_choker_limit_and_discard_blocked_card() -> None:
    parent = _parent({"play_top_card": 2})
    children = tuple(
        CardInstance(
            instance_id=f"child:{index}",
            card_id="child",
            type="attack",
            target="enemy",
            effects={"damage": 6},
            exhausts=True,
        )
        for index in range(2)
    )
    state = _with_cards(_combat(parent, metadata={"turn_card_play_limit": 6}), draw=children)
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "cards_played_this_turn": tuple(f"earlier:{index}" for index in range(5)),
                }
            )
        }
    )
    state = _play(state)
    assert state.combat.monsters[0].hp == 94
    assert len(state.combat.cards_played_this_turn) == 7  # Parent resolves after its children.
    assert [card.instance_id for card in state.combat.exhaust_pile] == ["child:0"]
    assert sorted(card.instance_id for card in state.combat.discard_pile) == ["child:1", "glam:1"]
    assert sum(event.kind == "card_autoplay_blocked" for event in state.replay_log[-1].events) == 1


def test_unplayable_top_card_is_discarded_without_effects_or_play_hooks() -> None:
    child = CardInstance(
        instance_id="child",
        card_id="child",
        type="status",
        target="enemy",
        effects={"damage": 50},
        custom={"unplayable": True},
    )
    state = _play(_with_cards(_combat(_parent({"play_top_card": 1})), draw=(child,)))
    assert state.combat.monsters[0].hp == 100
    assert len(state.combat.cards_played_this_turn) == 1
    assert sorted(card.instance_id for card in state.combat.discard_pile) == ["child", "glam:1"]


def test_sly_chosen_discard_runs_glam_play_hooks_and_exhausts_physical_card_once() -> None:
    parent = _parent({"discard_choice": 1})
    sly = CardInstance(
        instance_id="sly",
        card_id="sly",
        type="skill",
        target="self",
        cost=9,
        effects={"block": 1},
        exhausts=True,
        custom={"sly": True, "star_cost": 9},
        enchantments=(CardEnchantment(keyword="glam", amount=1),),
    )
    state = _player_statuses(_with_cards(_combat(parent), hand=(parent, sly)), afterimage=1)
    state = _choose(_play(state), "discard_card", "sly")
    assert state.player.block == 5
    assert state.player.energy == 3
    assert state.player.resources["star"] == 3
    assert state.combat.metadata["cards_played_this_combat"] == 3
    assert len(state.combat.cards_played_this_turn) == 3
    assert [card.instance_id for card in state.combat.exhaust_pile] == ["sly"]
    assert [card.instance_id for card in state.combat.discard_pile] == ["glam:1"]
    assert not state.combat.pending_choices


def test_random_discard_sly_exhaust_triggers_once_through_enclosing_effect() -> None:
    parent = _parent({"discard_random": 1})
    sly = CardInstance(
        instance_id="sly",
        card_id="sly",
        type="attack",
        target="enemy",
        effects={"damage": 6},
        exhausts=True,
        custom={"sly": True},
    )
    state = _with_cards(_combat(parent, relics=("charons_ashes",)), hand=(parent, sly))
    state = _play(state)
    assert state.combat.monsters[0].hp == 91  # Six attack and one three-damage exhaust trigger.
    assert len(state.combat.exhaust_pile) == 1
    assert state.combat.metadata["cards_played_this_combat"] == 2


def test_sly_lethal_choice_finishes_combat_instead_of_leaving_empty_combat() -> None:
    parent = _parent({"discard_choice": 1})
    sly = CardInstance(
        instance_id="sly",
        card_id="sly",
        type="attack",
        target="enemy",
        effects={"damage": 6},
        custom={"sly": True},
    )
    state = _with_cards(_combat(parent, enemy_hp=(5,)), hand=(parent, sly))
    state = _choose(_play(state), "discard_card", "sly")
    assert state.phase != RunPhase.COMBAT
    assert any(event.kind == "monster_defeated" for event in state.replay_log[-1].events)


def test_chosen_play_three_times_runs_hooks_and_moves_exhaust_card_once() -> None:
    parent = _parent(
        {"choose_card": {"action": "play", "zone": "hand", "count": 1, "play_times": 3}}
    )
    chosen = CardInstance(
        instance_id="chosen",
        card_id="chosen",
        type="skill",
        target="self",
        cost=9,
        effects={"block": 2},
        exhausts=True,
        custom={"star_cost": 9},
    )
    state = _player_statuses(
        _with_cards(
            _combat(parent, relics=("charons_ashes",)),
            hand=(parent, chosen),
        ),
        afterimage=1,
    )
    state = _choose(_play(state), "choose_card", "chosen")
    assert state.player.block == 10
    assert state.player.energy == 3
    assert state.player.resources["star"] == 3
    assert state.combat.metadata["cards_played_this_combat"] == 4
    assert len(state.combat.cards_played_this_turn) == 4
    assert state.combat.monsters[0].hp == 97
    assert [card.instance_id for card in state.combat.exhaust_pile] == ["chosen"]
    assert [card.instance_id for card in state.combat.discard_pile] == ["glam:1"]
    assert sum(event.kind == "card_played_by_choice" for event in state.replay_log[-1].events) == 3


def test_chosen_repeats_wait_for_their_own_modal_choices_after_reload() -> None:
    parent = _parent(
        {"choose_card": {"action": "play", "zone": "hand", "count": 1, "play_times": 3}}
    )
    chosen = CardInstance(
        instance_id="chosen",
        card_id="chosen",
        type="skill",
        target="self",
        effects={"block": 1, "discard_choice": 1},
    )
    junk = tuple(CardInstance(instance_id=f"junk:{index}", card_id="junk") for index in range(3))
    state = _with_cards(_combat(parent), hand=(parent, chosen) + junk)
    state = _choose(_play(state), "choose_card", "chosen")
    assert state.player.block == 1
    assert len(state.combat.metadata["card_replay_queue"]) == 2
    for index in range(3):
        state = load_state(json.loads(json.dumps(serialize(state))))
        state = _choose(state, "discard_card", f"junk:{index}")
        assert state.player.block == min(3, index + 2)
    assert not state.combat.pending_choices
    assert not state.combat.metadata.get("card_replay_queue")
    assert state.combat.metadata["cards_played_this_combat"] == 4
    assert len({card.instance_id for card in state.combat.discard_pile}) == 5


def test_nested_autoplay_kill_rewards_gremlin_horn_once() -> None:
    parent = _parent({"play_top_card": 1})
    child = CardInstance(
        instance_id="child",
        card_id="child",
        type="attack",
        target="enemy",
        effects={"damage": 6},
    )
    draw = CardInstance(instance_id="draw", card_id="draw")
    state = _with_cards(
        _combat(parent, enemy_hp=(5, 100), relics=("gremlin_horn",)), draw=(child, draw)
    )
    state = _play(state)
    assert state.player.energy == 4
    assert [card.instance_id for card in state.combat.hand] == ["draw"]
    assert state.combat.monsters[0].hp == 0
    assert state.combat.monsters[1].hp == 100


def test_multiple_top_autoplays_wait_for_child_choice_and_survive_reload() -> None:
    parent = _parent({"play_top_card": 2})
    first = CardInstance(
        instance_id="first",
        card_id="first",
        type="skill",
        target="self",
        effects={"block": 1, "discard_choice": 1},
    )
    second = CardInstance(
        instance_id="second",
        card_id="second",
        type="skill",
        target="self",
        effects={"block": 10},
    )
    junk = CardInstance(instance_id="junk", card_id="junk")
    state = _with_cards(_combat(parent), hand=(parent, junk), draw=(first, second))
    state = _play(state)
    assert state.player.block == 1
    assert [card.instance_id for card in state.combat.draw_pile] == ["second"]
    assert len(state.combat.metadata["card_replay_queue"]) == 1
    state = load_state(json.loads(json.dumps(serialize(state))))
    state = _choose(state, "discard_card", "junk")
    assert state.player.block == 11
    assert state.combat.metadata["cards_played_this_combat"] == 3
    assert not state.combat.draw_pile
    assert not state.combat.pending_choices
    assert not state.combat.metadata.get("card_replay_queue")


def test_random_hand_trigger_uses_same_free_autoplay_hooks() -> None:
    parent = _parent({"block": 1})
    child = CardInstance(
        instance_id="child",
        card_id="child",
        type="attack",
        target="enemy",
        cost=9,
        custom={"star_cost": 9},
        effects={"damage": 6},
    )
    trigger = {
        "trigger": "card_played",
        "source_card_id": "listener",
        "condition": {"card_type": "skill"},
        "effects": ({"play_random_card_from_hand": {"card_type": "attack"}},),
    }
    state = _player_statuses(
        _with_cards(
            _combat(parent, metadata={"combat_triggers": (trigger,)}),
            hand=(parent, child),
        ),
        afterimage=1,
    )
    state = _play(state)
    assert state.player.block == 3
    assert state.player.energy == 3
    assert state.player.resources["star"] == 3
    assert state.combat.monsters[0].hp == 94
    assert state.combat.metadata["cards_played_this_combat"] == 2
    assert state.combat.metadata["attacks_played_this_turn"] == 1
    assert sorted(card.instance_id for card in state.combat.discard_pile) == ["child", "glam:1"]
