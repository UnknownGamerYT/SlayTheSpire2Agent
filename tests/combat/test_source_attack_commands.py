"""Exercise source AfterAttack boundaries through public actions and reloads."""
import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _play
from combat.test_stable_item_contracts import _enemies, _potion


def test_gigantification_and_vigor_consume_after_first_command_before_second_command():
    attack = _simple("two_commands", card_type="attack", target="enemy", effects={
        "sequence": ({"damage": 5}, {"damage": 5}),
    })
    state = _potion(_state(attack, statuses={"vigor": 2}), "gigantification_potion")
    result = _play(_reload(state), "two_commands")
    assert result.combat.monsters[0].hp == 74  # (5+2)*3, then 5.
    assert not result.player.statuses.get("vigor")
    assert not result.combat.metadata.get("next_card_damage_multiplier")
    damage = [e.amount for e in result.replay_log[-1].events if e.kind == "monster_damaged"]
    assert damage == [21, 5]


def test_twin_strike_source_hit_count_keeps_modifiers_for_both_hits():
    state = _potion(_state(_source("TWIN_STRIKE"), statuses={"vigor": 2}),
                    "gigantification_potion")
    result = _play(_reload(state), "twin_strike")
    damage = [e.amount for e in result.replay_log[-1].events if e.kind == "monster_damaged"]
    assert damage == [21, 21]
    assert result.combat.monsters[0].hp == 58
    assert not result.combat.metadata.get("source_attack_commands")


def test_dagger_spray_source_all_enemy_batches_keep_modifiers_for_every_hit_and_target():
    state = _potion(_enemies(_state(_source("DAGGER_SPRAY"), statuses={"vigor": 2})),
                    "gigantification_potion")
    result = _play(_reload(state), "dagger_spray")
    assert [m.hp for m in result.combat.monsters] == [64, 64]
    events = result.replay_log[-1].events
    assert [e.amount for e in events if e.kind == "monster_damaged"] == [18] * 4
    assert sum(e.kind == "next_card_damage_multiplier_consumed" for e in events) == 1


def test_zero_hit_source_barrage_still_finishes_attack_command():
    state = _potion(_state(_source("BARRAGE"), statuses={"vigor": 2}),
                    "gigantification_potion")
    result = _play(_reload(state), "barrage")
    assert result.combat.monsters[0].hp == 100
    assert not result.player.statuses.get("vigor")
    assert not result.combat.metadata.get("next_card_damage_multiplier")


def test_missing_osty_has_no_attack_command_and_preserves_owner_modifiers():
    state = _potion(_state(_source("UNLEASH"), statuses={"vigor": 2}),
                    "gigantification_potion")
    result = _play(_reload(state), "unleash")
    assert result.combat.monsters[0].hp == 100
    assert any(e.kind == "osty_action_blocked" for e in result.replay_log[-1].events)
    assert result.player.statuses["vigor"] == 2
    assert result.combat.metadata.get("next_card_damage_multiplier")


@pytest.mark.parametrize("card_id,hits,base", [
    ("ASTRAL_PULSE", 2, 6), ("CELESTIAL_MIGHT", 3, 6), ("CONFLAGRATION", 4, 2),
    ("DAGGER_SPRAY", 2, 4), ("EXTERMINATE", 4, 3), ("FIGHT_ME", 2, 5),
    ("GUNK_UP", 3, 4), ("MAUL", 2, 5), ("PECK", 3, 2), ("REFRACT", 2, 9),
    ("SEVEN_STARS", 7, 7), ("TWIN_STRIKE", 2, 5), ("UPROAR", 2, 6),
])
def test_stable_static_hit_commands_keep_gigantification_for_full_hit_count(card_id, hits, base):
    state = _potion(_enemies(_state(_source(card_id)), hp=1000), "gigantification_potion")
    result = _play(_reload(state), card_id.lower())
    amounts = [e.amount for e in result.replay_log[-1].events if e.kind == "monster_damaged"
               and e.target_id == "target"]
    assert amounts == [base * 3] * hits
    assert not result.combat.metadata.get("next_card_damage_multiplier")


def test_uproar_finishes_own_command_before_autoplaying_child_attack():
    child = _simple("child", card_type="attack", target="enemy", effects={"damage": 5})
    state = _potion(_state(_source("UPROAR"), statuses={"vigor": 2}, draw=(child,)),
                    "gigantification_potion")
    result = _play(_reload(state), "uproar")
    assert result.combat.monsters[0].hp == 47  # (6+2)*3 twice, then child 5.
    assert [e.amount for e in result.replay_log[-1].events if e.kind == "monster_damaged"] == [
        24, 24, 5,
    ]


def test_combat_ending_skips_later_commands_and_preserves_second_queued_charge():
    attack = _simple("two_commands", card_type="attack", target="enemy", effects={
        "sequence": ({"damage": 40}, {"damage": 5}),
    })
    state = _potion(_potion(_state(attack), "gigantification_potion"), "gigantification_potion")
    result = _play(_reload(state), "two_commands")
    events = result.replay_log[-1].events
    assert sum(e.kind == "next_card_damage_multiplier_consumed" for e in events) == 1
