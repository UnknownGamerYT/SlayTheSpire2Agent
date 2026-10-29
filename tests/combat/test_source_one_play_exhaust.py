"""CardModel.ExhaustOnNextPlay is consumed once and does not add a keyword."""
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play

from sts2sim.engine import CardEnchantment


def test_one_play_exhaust_survives_reload_then_returned_card_discards_normally():
    card = _simple("card", card_type="attack", target="enemy", effects={"damage": 1},
                   custom={"source_exhaust_on_next_play": True})
    state = _play(_reload(_state(card)), "card")
    exhausted = state.combat.exhaust_pile[0]
    assert not exhausted.exhausts
    assert not exhausted.custom.get("source_exhaust_on_next_play")
    combat = state.combat.model_copy(update={"hand": (exhausted,), "exhaust_pile": ()})
    state = _play(_reload(state.model_copy(update={"combat": combat})), "card")
    assert not state.combat.exhaust_pile
    assert state.combat.discard_pile[0].instance_id == "card"


def test_replayed_card_has_one_physical_exhaust_without_changing_replay_count():
    card = _simple("card", card_type="attack", target="enemy", effects={"damage": 1},
                   enchantments=(CardEnchantment(keyword="replay", amount=2),),
                   custom={"source_exhaust_on_next_play": True})
    state = _play(_state(card), "card")
    assert state.combat.monsters[0].hp == 97
    assert len(state.combat.exhaust_pile) == 1
    assert sum(e.kind == "card_exhausted" for e in state.combat.last_events) == 1


def test_power_card_removal_has_source_priority_over_one_play_exhaust():
    card = _simple("card", card_type="power", custom={"source_exhaust_on_next_play": True})
    state = _play(_state(card), "card")
    assert not state.combat.hand
    assert not state.combat.discard_pile
    assert not state.combat.exhaust_pile
