from __future__ import annotations

import json

from combat.test_glam import _card, _combat, _end_turn, _play

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardInstance, CardType


def _player_statuses(state, **statuses):
    player = state.player.model_copy(update={"statuses": statuses})
    return state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )


def test_glam_is_two_card_plays_for_afterimage_and_play_counters() -> None:
    state = _play(_player_statuses(_combat(), afterimage=3))
    assert state.player.block == 6
    assert len(state.combat.cards_played_this_turn) == 2
    assert state.combat.metadata["cards_played_this_combat"] == 2
    assert state.combat.metadata["attacks_played_this_turn"] == 2
    assert state.player.energy == 2
    assert state.player.resources["star"] == 2
    assert [card.instance_id for card in state.combat.discard_pile] == ["glam:1"]


def test_glam_updates_slow_and_withering_presence_once_per_play() -> None:
    state = _combat(_card(effects={"damage": 10}))
    monster = state.combat.monsters[0].model_copy(
        update={"statuses": {"slow": 1, "withering_presence": 2}}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state)
    assert state.combat.monsters[0].hp == 77  # 11 damage, then 12 after the second Slow trigger.
    assert state.combat.monsters[0].metadata["slow_cards_played"] == 2
    assert len([card for card in state.combat.hand if card.card_id == "wither"]) == 1


def test_glam_skill_triggers_enrage_and_vital_spark_for_both_plays() -> None:
    card = _card(effects={"block": 1}, target="self").model_copy(update={"type": CardType.SKILL})
    state = _combat(card)
    monster = state.combat.monsters[0].model_copy(
        update={"statuses": {"enrage": 2, "vital_spark": 1}}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state)
    assert state.combat.monsters[0].statuses["strength"] == 4
    assert state.player.statuses["tainted"] == 2
    assert state.combat.metadata["skills_played_this_turn"] == 2


def test_glam_counts_toward_attack_relic_thresholds_without_repaying_energy() -> None:
    state = _combat(
        relics=("kunai", "shuriken", "nunchaku"),
        metadata={"attacks_played_this_turn": 1, "relic_counters": {"nunchaku": 9}},
    )
    state = _play(state)
    assert state.player.statuses["dexterity"] == 1
    assert state.player.statuses["strength"] == 1
    assert state.player.energy == 3  # Pay once, gain Nunchaku's one energy.
    assert state.combat.metadata["relic_counters"]["nunchaku"] == 1


def test_throwing_axe_and_glam_each_add_one_play_without_recursion() -> None:
    state = _player_statuses(_combat(relics=("throwing_axe", "shuriken")), afterimage=2)
    state = _play(state)
    assert state.combat.monsters[0].hp == 82
    assert state.player.block == 6
    assert state.player.statuses["strength"] == 1
    assert len(state.combat.cards_played_this_turn) == 3
    assert state.combat.metadata["cards_played_this_combat"] == 3
    assert len(state.combat.discard_pile) == 1
    assert not state.combat.metadata.get("card_replay_queue")


def test_first_card_replay_stays_with_original_when_it_autoplays_another_card() -> None:
    card = _card(effects={"play_top_card": 1}, target="self").model_copy(
        update={"type": CardType.SKILL, "enchantments": ()}
    )
    state = _combat(card, relics=("throwing_axe",))
    child = CardInstance(
        instance_id="child", card_id="child", type="attack", target="enemy", effects={"damage": 6}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_pile": (child,)})}
    )
    state = _play(state)
    axe_replays = [
        event
        for event in state.replay_log[-1].events
        if event.kind == "card_extra_played" and event.metadata.get("relic_id") == "throwing_axe"
    ]
    assert len(axe_replays) == 1
    assert axe_replays[0].target_id == "glam:1"
    assert state.combat.monsters[0].hp == 94
    assert state.combat.metadata["cards_played_this_combat"] == 3


def test_throwing_axe_replays_once_per_combat_and_does_not_rearm_next_turn() -> None:
    state = _play(_combat(relics=("throwing_axe",)))
    assert state.combat.monsters[0].hp == 82
    state = _play(_end_turn(state))
    assert state.combat.monsters[0].hp == 76
    assert len(state.combat.cards_played_this_turn) == 1


def test_sixth_card_replay_resolves_beyond_choker_limit_but_blocks_new_manual_cards() -> None:
    state = _combat(metadata={"turn_card_play_limit": 6, "cards_played_this_combat": 5})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "cards_played_this_turn": tuple(f"earlier:{index}" for index in range(5)),
                }
            ),
        }
    )
    state = _play(state)
    assert len(state.combat.cards_played_this_turn) == 7
    assert state.combat.monsters[0].hp == 88
    assert not any(action.type == "play_card" for action in legal_actions(state))


def test_double_tap_and_glam_each_add_one_play_and_consume_modifier_once() -> None:
    state = _combat(
        metadata={
            "next_card_extra_play": (
                {"card_type": "attack", "amount": 1, "source_card_id": "double_tap"},
            ),
        }
    )
    state = _play(_player_statuses(state, afterimage=1))
    assert state.combat.monsters[0].hp == 82
    assert state.player.block == 3
    assert len(state.combat.cards_played_this_turn) == 3
    assert "next_card_extra_play" not in state.combat.metadata


def test_replay_keyword_repeats_n_times_with_full_hooks_without_recursion() -> None:
    card = _card().model_copy(update={"enchantments": (), "custom": {"replay": 2}})
    state = _play(_player_statuses(_combat(card), afterimage=1))
    assert state.combat.monsters[0].hp == 82
    assert state.player.block == 3
    assert len(state.combat.cards_played_this_turn) == 3
    assert state.combat.metadata["cards_played_this_combat"] == 3
    assert len(state.combat.discard_pile) == 1
    assert state.player.energy == 2
    assert not state.combat.metadata.get("card_replay_queue")


def test_one_shot_vigor_is_consumed_before_glam_replay() -> None:
    state = _play(_player_statuses(_combat(), vigor=5))
    assert state.combat.monsters[0].hp == 83  # 6+5 on the original, 6 on replay.
    assert "vigor" not in state.player.statuses


def test_replay_applies_self_cost_changes_to_same_physical_card_without_repaying() -> None:
    state = _play(_combat(_card(effects={"damage": 6, "self_cost_delta": 1})))
    assert state.player.energy == 2
    assert len(state.combat.discard_pile) == 1
    assert state.combat.discard_pile[0].cost == 3
    assert "played_card_updates" not in state.combat.metadata


def test_replayed_exhaust_card_moves_once_and_exhaust_trigger_fires_once() -> None:
    card = _card().model_copy(update={"exhausts": True})
    state = _player_statuses(_combat(card, relics=("charons_ashes",)), afterimage=1)
    drawn = CardInstance(instance_id="drawn", card_id="drawn", type="skill", target="self")
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_pile": (drawn,)})}
    )
    state = _play(state)
    assert state.player.block == 2
    assert len(state.combat.exhaust_pile) == 1
    assert not state.combat.hand
    assert [card.instance_id for card in state.combat.draw_pile] == ["drawn"]
    assert state.combat.monsters[0].hp == 85
    assert sum(event.kind == "card_exhausted" for event in state.replay_log[-1].events) == 1
    assert not state.combat.discard_pile


def test_potion_generation_card_generates_potion_on_each_play() -> None:
    card = _card(effects={"add_random_potion": 1}, target="self").model_copy(
        update={"type": CardType.SKILL}
    )
    state = _combat(card)
    state = state.model_copy(update={"flags": {**state.flags, "potion_pool": ("fire_potion",)}})
    state = _play(state)
    assert state.potions == ("fire_potion", "fire_potion")
    assert "replay_potions" not in state.combat.metadata
    assert state.player.resources["star"] == 2


def test_glam_replay_waits_for_choices_and_resumes_after_json_reload() -> None:
    card = _card(
        effects={
            "block": 2,
            "choose_card": {"action": "discard", "zone": "hand", "count": 1},
        },
        target="self",
    ).model_copy(update={"type": CardType.SKILL})
    state = _combat(card)
    junk = tuple(
        CardInstance(instance_id=f"junk:{index}", card_id="junk", type="skill", target="self")
        for index in range(2)
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"hand": (card,) + junk})}
    )
    state = _play(state)
    assert state.player.block == 2
    assert state.combat.pending_choices
    assert state.combat.metadata["card_replay_queue"]
    state = load_state(json.loads(json.dumps(serialize(state))))
    state = step(
        state, next(action for action in legal_actions(state) if action.type == "discard_card")
    )
    assert state.player.block == 4
    assert state.combat.pending_choices  # The replay has its own discard choice.
    assert len(state.combat.cards_played_this_turn) == 2
    state = step(
        state, next(action for action in legal_actions(state) if action.type == "discard_card")
    )
    assert not state.combat.pending_choices
    assert not state.combat.metadata.get("card_replay_queue")
    assert len(state.combat.discard_pile) == 3


def test_history_course_copy_uses_card_play_hooks_and_does_not_rearm_glam() -> None:
    state = _play(_player_statuses(_combat(relics=("history_course",)), afterimage=1))
    assert state.player.block == 2
    state = _end_turn(state)
    assert state.player.block == 1
    assert state.player.energy == 3
    assert len(state.combat.cards_played_this_turn) == 1
    assert state.combat.metadata["cards_played_this_combat"] == 3
    assert state.combat.monsters[0].hp == 82
