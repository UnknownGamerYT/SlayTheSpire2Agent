"""Source command-order fixtures; callback relic IDs are forced deliberately.

The custom bag contents exercise await/resume ordering, not natural pool rarity
membership. Each actual event callback is reached through public actions.
"""
import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import ActionType, RoomKind, RunPhase
from sts2sim.engine import transitions as engine
from sts2sim.engine.source_native_event_completion import PENDING, has_obtain_commands
from sts2sim.mechanics.source_event_rules import SOURCE_EVENT_RULES


def _event(identifier, *, page="INITIAL", relics=(), potions=(), bag_first="CAULDRON"):
    state = new_run("native-await", "IRONCLAD", 0)
    state = state.model_copy(update={
        "relics": state.relics + tuple(relics), "potions": tuple(potions),
        "player": state.player.model_copy(update={"gold": 1000}),
        "flags": {**state.flags, "event_id": identifier, "event_flow_page_id": page,
            "player_relic_bag": {rarity: (bag_first, "ANCHOR")
                                 for rarity in ("Common", "Uncommon", "Rare")}},
    })
    return engine._enter_event_room(state, state.map.nodes[0].model_copy(
        update={"kind": RoomKind.EVENT}),
    )[0]


def _act(state, kind, target=None):
    action = next(a for a in legal_actions(state) if a.type == kind
                  and (target is None or str(a.target_id).casefold() == target.casefold()))
    before = serialize(state)
    result = step(state, action)
    assert serialize(state) == before
    assert serialize(result) == serialize(step(load_state(before), action))
    return result


def _draws(state, name="rewards"):
    stream = state.rng.streams.get(name)
    return stream.draws if stream else 0


def test_trial_waits_for_first_callback_before_pulling_second_relic():
    state = _event("TRIAL", page="MERCHANT", relics=("bing_bong", "darkstone_periapt"))
    before = _draws(state)
    hp = state.player.max_hp
    state = _act(state, ActionType.CHOOSE_EVENT, "GUILTY")
    assert state.phase is RunPhase.REWARD and state.reward.reward_id == "pickup:cauldron"
    assert "cauldron" in state.relics and "anchor" not in state.relics
    assert state.event.resolved_option_id is None
    assert state.flags[PENDING]["cursor"] == 1
    assert _draws(state) == before + 11  # First rarity + five two-draw potions.
    assert sum(card.card_id == "regret" for card in state.master_deck) == 2
    assert state.player.max_hp == hp + 12  # Curse entry and its Bing clone settle first.
    state = _act(state, ActionType.PROCEED)
    assert state.phase is RunPhase.EVENT and state.event.resolved_option_id == "GUILTY"
    assert state.relics.count("cauldron") == state.relics.count("anchor") == 1
    assert _draws(state) == before + 12
    assert PENDING not in state.flags and "source_pickup_reward_stack" not in state.flags
    assert sum(card.card_id == "regret" for card in state.master_deck) == 2
    assert state.player.max_hp == hp + 12
    assert sum(event.kind == "event_option_chosen" for entry in state.replay_log
               for event in entry.events) == 1


def test_mid_callback_potion_rng_is_preserved_before_the_next_relic_pull():
    state = _event("TRIAL", page="MERCHANT", potions=("entropic_brew",))
    state = _act(state, ActionType.CHOOSE_EVENT, "GUILTY")
    before_use = _draws(state, "combat_potion_generation")
    state = _act(state, ActionType.USE_POTION)
    assert _draws(state, "combat_potion_generation") > before_use
    before_resume = _draws(state)
    assert "anchor" not in state.relics
    state = _act(state, ActionType.PROCEED)
    assert state.relics.count("anchor") == 1 and _draws(state) == before_resume + 1


def test_ornate_curse_is_added_only_after_the_relic_deck_choice_finishes():
    state = _event("THIS_OR_THAT", bag_first="DOLLYS_MIRROR")
    before = len(state.master_deck)
    state = _act(state, ActionType.CHOOSE_EVENT, "ORNATE")
    assert "dollys_mirror" in state.relics
    assert not any(card.card_id == "clumsy" for card in state.master_deck)
    assert state.event.resolved_option_id is None and state.flags[PENDING]["cursor"] == 1
    assert {a.type for a in legal_actions(state)} == {ActionType.CHOOSE_CARD}
    state = _act(state, ActionType.CHOOSE_CARD)
    assert len(state.master_deck) == before + 2
    assert sum(card.card_id == "clumsy" for card in state.master_deck) == 1
    assert state.event.resolved_option_id.upper() == "ORNATE" and PENDING not in state.flags


def test_wongo_points_and_badge_wait_for_the_purchased_relic_callback():
    state = _event("WELCOME_TO_WONGOS")
    state = state.model_copy(update={"flags": {**state.flags, "wongo_points": 1990,
        "player_relic_bag": {"Common": ("CAULDRON",), "Rare": ("ANCHOR",),
                             "Uncommon": ("ANCHOR",)}}})
    state = _act(state, ActionType.CHOOSE_EVENT, "BARGAIN_BIN")
    assert state.phase is RunPhase.REWARD and state.player.gold == 900
    assert state.flags["wongo_points"] == 1990
    assert "wongo_customer_appreciation_badge" not in state.relics
    state = _act(state, ActionType.PROCEED)
    assert state.flags["wongo_points"] == 1990 and state.flags["wongo_points_earned"] == 32
    assert state.relics.count("wongo_customer_appreciation_badge") == 1
    assert state.event.resolved_option_id == "BARGAIN_BIN" and PENDING not in state.flags


@pytest.mark.parametrize("event_id", ["THE_FUTURE_OF_POTIONS", "POTION_COURIER", "BRAIN_LEECH"])
def test_native_custom_rewards_open_immediately_and_finish_after_child_rewards(event_id):
    state = _event(event_id, potions=("fruit_juice", "blood_potion"))
    target = {"POTION_COURIER": "GRAB_POTIONS", "BRAIN_LEECH": "RIP"}.get(event_id)
    state = _act(state, ActionType.CHOOSE_EVENT, target)
    assert state.phase is RunPhase.REWARD
    assert state.event.resolved_option_id is None and PENDING in state.flags
    if event_id == "THE_FUTURE_OF_POTIONS":
        assert state.flags["can_remove_potions"] is False
        assert not any(a.type in {ActionType.USE_POTION, ActionType.DISCARD_POTION}
                       for a in legal_actions(state))
    # Custom rewards permit explicit skip/close; their callback then finishes.
    for _ in range(12):
        actions = legal_actions(state)
        proceed = next((a for a in actions if a.type is ActionType.PROCEED), None)
        if proceed is not None:
            state = _act(state, ActionType.PROCEED)
            break
        state = _act(state, ActionType.SKIP_REWARD)
    assert state.phase is RunPhase.EVENT and state.event.resolved_option_id is not None
    assert PENDING not in state.flags
    if event_id == "THE_FUTURE_OF_POTIONS":
        assert "can_remove_potions" not in state.flags
        assert any(a.type is ActionType.USE_POTION for a in legal_actions(state))


def test_current_native_relic_callbacks_have_no_unqueued_generated_reward_tail():
    # A new source rule that combines these awaits must extend the serialized
    # continuation rather than silently dropping a future generated reward.
    for rules in SOURCE_EVENT_RULES.values():
        for rule in rules.values():
            payload = rule.payload()
            if has_obtain_commands(payload):
                assert not (rule.generated_count or rule.fixed_potion_rewards
                            or rule.random_potion_rewards or rule.uniform_potion_reward)


def test_last_custom_card_reward_closes_and_unlocks_potions_in_the_same_action():
    state = _event("THE_FUTURE_OF_POTIONS", potions=("fruit_juice", "blood_potion"))
    state = _act(state, ActionType.CHOOSE_EVENT)
    assert state.phase is RunPhase.REWARD and state.flags["can_remove_potions"] is False
    state = _act(state, ActionType.TAKE_REWARD_CARD)
    assert state.phase is RunPhase.EVENT and state.reward is None
    assert state.event.resolved_option_id is not None
    assert PENDING not in state.flags and "can_remove_potions" not in state.flags
    assert len(state.potions) == 1
    assert any(a.type is ActionType.USE_POTION for a in legal_actions(state))


def test_ranwid_keeps_potions_locked_until_an_awaited_relic_reward_finishes():
    state = _event("RANWID_THE_ELDER", potions=("fruit_juice", "blood_potion"))
    state = _act(state, ActionType.CHOOSE_EVENT, "GOLD")
    assert state.phase is RunPhase.REWARD and state.flags["can_remove_potions"] is False
    assert state.event.resolved_option_id is None
    assert not any(a.type in {ActionType.USE_POTION, ActionType.DISCARD_POTION}
                   for a in legal_actions(state))
    state = _act(state, ActionType.PROCEED)
    assert state.phase is RunPhase.EVENT and state.event.resolved_option_id.upper() == "GOLD"
    assert "can_remove_potions" not in state.flags
    assert any(a.type is ActionType.USE_POTION for a in legal_actions(state))


def test_completion_preserves_an_existing_outer_potion_removal_lock():
    state = _event("THE_FUTURE_OF_POTIONS", potions=("fruit_juice", "blood_potion"))
    state = state.model_copy(update={"flags": {**state.flags, "can_remove_potions": False}})
    state = _act(state, ActionType.CHOOSE_EVENT)
    state = _act(state, ActionType.PROCEED)
    assert state.phase is RunPhase.EVENT and state.flags["can_remove_potions"] is False
    assert PENDING not in state.flags
    assert not any(a.type in {ActionType.USE_POTION, ActionType.DISCARD_POTION}
                   for a in legal_actions(state))


def test_colorful_philosophers_waits_for_all_three_custom_card_rewards():
    state = _event("COLORFUL_PHILOSOPHERS")
    state = _act(state, ActionType.CHOOSE_EVENT)
    assert state.phase is RunPhase.REWARD and state.event.resolved_option_id is None
    assert len(state.reward.card_option_groups) == 2
    for remaining in (2, 1, 0):
        state = _act(state, ActionType.TAKE_REWARD_CARD)
        if remaining:
            assert state.phase is RunPhase.REWARD and PENDING in state.flags
        else:
            assert state.phase is RunPhase.EVENT and state.reward is None
            assert PENDING not in state.flags and state.event.resolved_option_id is not None
