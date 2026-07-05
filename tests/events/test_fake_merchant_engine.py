from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import EventOptionState, EventState, RunPhase


def _first_action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type and (target_id is None or action.target_id == target_id)
    )


def test_foul_potion_starts_fake_merchant_combat_and_rewards_unsold_relics() -> None:
    state = new_run(
        seed=9801,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.EVENT,
            "potions": ("foul_potion",),
            "event": EventState(
                event_id="fake_merchant",
                name="Fake Merchant",
                options=(
                    EventOptionState(
                        option_id="SUMMARY",
                        title="Fake Merchant Summary",
                        metadata={
                            "summary_marker": True,
                            "post_combat_reward": {
                                "fixed_relic_ids": (
                                    "fake_merchants_rug",
                                    "fake_anchor",
                                    "fake_mango",
                                ),
                            },
                        },
                    ),
                ),
            ),
        }
    )

    assert not any(action.type == "choose_event" for action in legal_actions(state))
    throw = _first_action(state, "throw_potion_at_merchant", "fake_merchant")

    state = step(state, throw)

    assert state.phase is RunPhase.COMBAT
    assert state.potions == ()
    assert state.flags["fake_merchant_combat"] is True
    assert state.flags["fake_merchant_unsold_relic_ids"] == ("fake_anchor", "fake_mango")
    assert state.replay_log[-1].events[0].kind == "foul_potion_thrown_at_fake_merchant"

    state = step(state, _first_action(state, "play_card"))

    assert state.phase is RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.gold == 0
    assert state.reward.card_options == ()
    assert state.reward.relic_ids == (
        "fake_merchants_rug",
        "fake_anchor",
        "fake_mango",
    )

