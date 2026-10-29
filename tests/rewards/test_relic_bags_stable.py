from __future__ import annotations

import pytest

from sts2sim.mechanics.relic_bags import (
    RelicBagContext,
    pull_relic_from_bag,
    relic_is_allowed,
    remove_bag_relics,
)


def test_front_back_and_no_replacement() -> None:
    bag = {"Common": ["ANCHOR", "LANTERN", "BREAD"]}
    front = pull_relic_from_bag(bag, "Common", context=RelicBagContext())
    back = pull_relic_from_bag(front.bag, "Common", context=RelicBagContext(), from_back=True)
    assert front.relic_id == "ANCHOR"
    assert back.relic_id == "BREAD"
    assert back.bag == {"Common": ("LANTERN",)}
    assert bag == {"Common": ["ANCHOR", "LANTERN", "BREAD"]}


def test_shop_filter_skips_without_removing_and_uses_source_fallback() -> None:
    bag = {"Shop": [], "Common": ["OLD_COIN"], "Uncommon": ["THE_COURIER", "ANCHOR"]}
    result = pull_relic_from_bag(
        bag,
        "Shop",
        context=RelicBagContext(),
        for_shop=True,
        from_back=True,
    )
    assert result.relic_id == "ANCHOR"
    assert result.bag["Common"] == ("OLD_COIN",)
    assert result.bag["Uncommon"] == ("THE_COURIER",)
    reward = pull_relic_from_bag(result.bag, "Common", context=RelicBagContext())
    assert reward.relic_id == "OLD_COIN"


@pytest.mark.parametrize("players,threshold", [(1, 41), (2, 38)])
def test_late_relic_boundary_permanently_prunes_all_buckets(players: int, threshold: int) -> None:
    assert relic_is_allowed(
        "OLD_COIN", RelicBagContext(total_floor=threshold - 1, player_count=players)
    )
    bag = {"Common": ["BREAD"], "Rare": ["OLD_COIN", "FROZEN_EGG"]}
    result = pull_relic_from_bag(
        bag,
        "Common",
        context=RelicBagContext(total_floor=threshold, player_count=players),
    )
    assert result.relic_id == "BREAD"
    assert result.bag["Rare"] == ()


def test_rare_does_not_downgrade_and_exhaustion_is_explicit() -> None:
    result = pull_relic_from_bag({"Common": ["ANCHOR"]}, "Rare", context=RelicBagContext())
    assert result.relic_id is None
    assert result.bag["Common"] == ("ANCHOR",)


def test_refresh_preserves_original_source_order_and_does_not_touch_other_rarities() -> None:
    result = pull_relic_from_bag(
        {"Common": [], "Rare": []},
        "Common",
        context=RelicBagContext(),
        refresh_originals={"Common": ["ANCHOR", "BREAD"], "Rare": ["GIRYA"]},
    )
    assert result.relic_id == "ANCHOR"
    assert result.bag == {"Common": ("BREAD",), "Rare": ()}


def test_multiplayer_fallback_is_consumed_and_filters_are_preserved() -> None:
    result = pull_relic_from_bag(
        {},
        "Rare",
        context=RelicBagContext(player_count=2),
        multiplayer_fallback=["WINGED_BOOTS", "BREAD", "ANCHOR"],
        excluded_ids=["BREAD"],
    )
    assert result.relic_id == "ANCHOR"
    assert result.multiplayer_fallback == ("BREAD",)


@pytest.mark.parametrize(
    "relic,players,allowed",
    [
        ("WINGED_BOOTS", 1, True),
        ("WINGED_BOOTS", 2, False),
        ("SILVER_CRUCIBLE", 1, True),
        ("SILVER_CRUCIBLE", 2, False),
        ("MASSIVE_SCROLL", 1, False),
        ("MASSIVE_SCROLL", 2, True),
    ],
)
def test_source_multiplayer_predicates(relic: str, players: int, allowed: bool) -> None:
    assert relic_is_allowed(relic, RelicBagContext(player_count=players)) is allowed


def test_lasting_candy_first_run_ironclad_and_obtained_removal() -> None:
    assert not relic_is_allowed("LASTING_CANDY", RelicBagContext(first_run_ironclad_present=True))
    assert relic_is_allowed("LASTING_CANDY", RelicBagContext())
    assert remove_bag_relics(
        {"Common": ["BREAD", "ANCHOR"], "Rare": ["ANCHOR"]},
        ["anchor"],
    ) == {"Common": ("BREAD",), "Rare": ()}
