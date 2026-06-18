from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from helpers import project_root

from sts2sim import legal_actions, step


def _load_combat_reward_test() -> ModuleType:
    module_path = project_root() / "combat_reward_test.py"
    spec = importlib.util.spec_from_file_location("combat_reward_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["combat_reward_test"] = module
    spec.loader.exec_module(module)
    return module


combat_reward_test = _load_combat_reward_test()


def test_fake_merchant_page_payload_uses_multi_relic_reward() -> None:
    state = combat_reward_test.create_reward_state(
        seed=123,
        encounter="event",
        event_preset_id="fake_merchant",
        ascension=0,
        character_id="ironclad",
        fake_unsold_relics=("fake_anchor", "fake_mango"),
    )
    payload = combat_reward_test._state_payload(
        state,
        seed=123,
        encounter="event",
        event_preset_id="fake_merchant",
        message="ready",
    )

    assert payload["reward"]["relic_ids"] == [
        "fake_merchants_rug",
        "fake_anchor",
        "fake_mango",
    ]
    assert payload["reward"]["potion_id"] is None
    assert payload["reward"]["metadata"]["potion_drop"]["dropped"] is False


def test_event_catalog_includes_known_combat_events() -> None:
    event_ids = {
        event["event_id"] for event in combat_reward_test._event_catalog_payload()
    }

    assert {
        "BATTLEWORN_DUMMY",
        "DENSE_VEGETATION",
        "FAKE_MERCHANT",
        "THE_LANTERN_KEY",
    } <= event_ids


def test_punch_off_preset_models_greater_reward_bundle() -> None:
    state = combat_reward_test.create_reward_state(
        seed=125,
        encounter="event",
        event_preset_id="punch_off_greater_rewards",
        ascension=0,
        character_id="ironclad",
        fake_unsold_relics=(),
    )
    payload = combat_reward_test._state_payload(
        state,
        seed=125,
        encounter="event",
        event_preset_id="punch_off_greater_rewards",
        message="ready",
    )
    reward = payload["reward"]

    assert 10 <= reward["gold"] <= 20
    assert len(reward["card_options"]) == 3
    assert len(reward["relic_ids"]) == 1
    assert len(reward["potion_ids"]) == 1
    assert "{" not in reward["potion_ids"][0]
    assert reward["metadata"]["encounter"] == "normal"
    assert list(reward["metadata"]["extra_potion_ids"]) == reward["potion_ids"]


def test_dense_vegetation_preset_uses_standard_combat_rewards() -> None:
    state = combat_reward_test.create_reward_state(
        seed=127,
        encounter="event",
        event_preset_id="dense_vegetation",
        ascension=0,
        character_id="ironclad",
        fake_unsold_relics=(),
    )
    payload = combat_reward_test._state_payload(
        state,
        seed=127,
        encounter="event",
        event_preset_id="dense_vegetation",
        message="ready",
    )
    reward = payload["reward"]

    assert 10 <= reward["gold"] <= 20
    assert len(reward["card_options"]) == 3
    assert reward["relic_ids"] == []
    assert reward["metadata"]["encounter"] == "normal"


def test_lantern_key_preset_does_not_award_lantern_relic() -> None:
    state = combat_reward_test.create_reward_state(
        seed=126,
        encounter="event",
        event_preset_id="lantern_key",
        ascension=0,
        character_id="ironclad",
        fake_unsold_relics=(),
    )
    payload = combat_reward_test._state_payload(
        state,
        seed=126,
        encounter="event",
        event_preset_id="lantern_key",
        message="ready",
    )

    assert payload["reward"]["relic_ids"] == []
    assert payload["reward"]["card_ids"] == ["lantern_key"]
    assert len(payload["reward"]["card_options"]) == 3
    assert "lantern" not in payload["relics"]


def test_generated_reward_payload_can_claim_first_relic() -> None:
    state = combat_reward_test.create_reward_state(
        seed=124,
        encounter="event",
        event_preset_id="battleworn_dummy_relic",
        ascension=0,
        character_id="ironclad",
        fake_unsold_relics=(),
    )
    reward = state.reward
    assert reward is not None
    assert len(reward.relic_ids) == 1

    action = next(
        action
        for action in legal_actions(state)
        if action.type == "take_reward_relic"
    )
    state = step(state, action)

    assert reward.relic_ids[0] in state.relics
