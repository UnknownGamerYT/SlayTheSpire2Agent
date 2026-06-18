from __future__ import annotations

import importlib.util
from types import ModuleType

from helpers import project_root


def _load_shop_test() -> ModuleType:
    module_path = project_root() / "shop_test.py"
    spec = importlib.util.spec_from_file_location("shop_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shop_test = _load_shop_test()


def test_shop_payload_explains_full_potion_belt_block() -> None:
    state = shop_test.create_shop_state(
        seed=123,
        gold=350,
        ascension=0,
        relics=(),
        potions=("fire_potion", "skill_potion", "foul_potion"),
    )

    payload = shop_test._state_payload(state, seed=123, message="ready")
    potion_items = [item for item in payload["items"] if item["kind"] == "potion"]

    assert payload["potion_slots"] == 3
    assert payload["potion_slots_open"] == 0
    assert all(potion["can_discard"] for potion in payload["potions"])
    assert potion_items
    assert all(not item["can_buy"] for item in potion_items)
    assert {item["blocked_reason"] for item in potion_items} == {"Potion belt full"}
