from __future__ import annotations

import pytest

from sts2sim.mechanics.event_catalog import (
    event_catalog_coverage,
    event_catalog_ids,
    event_catalog_unsupported_categories,
    known_event_catalog_options,
)
from sts2sim.mechanics.event_rooms import EventOption


def _option(event_id: str, option_id: str) -> EventOption:
    for option in known_event_catalog_options(event_id):
        if option.option_id == option_id:
            return option
    raise AssertionError(f"Missing catalog option: {event_id} / {option_id}")


def _coverage(event_id: str, option_id: str):
    for marker in event_catalog_coverage(event_id):
        if marker.option_id == option_id:
            return marker
    raise AssertionError(f"Missing coverage row: {event_id} / {option_id}")


def test_catalog_ids_are_normalized_and_cover_many_events_across_acts() -> None:
    ids = event_catalog_ids()

    assert ids == tuple(sorted(ids))
    assert {
        "abyssal_baths",
        "amalgamator",
        "aroma_of_chaos",
        "battleworn_dummy",
        "bugslayer",
        "byrdonis_nest",
        "dense_vegetation",
        "drowning_beacon",
        "morphic_grove",
        "potion_courier",
        "punch_off",
        "round_tea_party",
        "tablet_of_truth",
        "the_lantern_key",
        "war_historian_repy",
        "waterlogged_scriptorium",
        "wellspring",
        "whispering_hollow",
        "wood_carvings",
        "zen_weaver",
    } <= set(ids)

    assert _option("THE_LANTERN_KEY", "return_the_key").option_id == "return_the_key"
    assert _option("repy", "unlock_cage").fixed_relic_ids == ("history_course",)


def test_hp_gold_and_max_hp_primitives_from_cached_text() -> None:
    immerse = _option("abyssal_baths", "immerse")
    abstain = _option("abyssal_baths", "abstain")
    byrdonis = _option("byrdonis_nest", "eat")
    trudge = _option("dense_vegetation", "trudge_on")
    rest = _option("dense_vegetation", "rest")
    bloody_ink = _option("waterlogged_scriptorium", "bloody_ink")

    assert immerse.max_hp_delta == 2
    assert immerse.hp_delta == -3
    assert abstain.hp_delta == 10
    assert byrdonis.max_hp_delta == 7
    assert trudge.hp_delta == -8
    assert trudge.metadata["gold_gain_range"] == (61, 99)
    assert rest.heal_percent_max_hp == pytest.approx(0.30)
    assert rest.combat_encounter == "normal"
    assert bloody_ink.max_hp_delta == 6


def test_fixed_card_add_remove_and_requirements_are_source_ids() -> None:
    strikes = _option("amalgamator", "combine_strikes")
    defends = _option("amalgamator", "combine_defends")
    bug = _option("bugslayer", "extermination")
    breathing = _option("zen_weaver", "breathing_techniques")
    lantern_cage = _option("war_historian_repy", "unlock_cage")

    assert strikes.fixed_card_ids == ("ultimate_strike",)
    assert strikes.remove_card_ids == ("strike_ironclad", "strike_ironclad")
    assert strikes.required_card_ids == ("strike_ironclad", "strike_ironclad")
    assert strikes.metadata["required_card_counts"] == {"strike_ironclad": 2}
    assert defends.fixed_card_ids == ("ultimate_defend",)
    assert defends.remove_card_ids == ("defend_ironclad", "defend_ironclad")
    assert bug.fixed_card_ids == ("exterminate",)
    assert breathing.gold_delta == -50
    assert breathing.fixed_card_ids == ("enlightenment", "enlightenment")
    assert lantern_cage.remove_card_ids == ("lantern_key",)
    assert lantern_cage.required_card_ids == ("lantern_key",)
    assert lantern_cage.fixed_relic_ids == ("history_course",)


def test_random_relic_potion_fixed_relic_and_fixed_potion_primitives() -> None:
    setting_1 = _option("battleworn_dummy", "setting_1")
    setting_3 = _option("battleworn_dummy", "setting_3")
    chest = _option("war_historian_repy", "unlock_chest")
    bottle = _option("drowning_beacon", "bottle")
    climb = _option("drowning_beacon", "climb")
    tea = _option("round_tea_party", "enjoy_tea")
    courier = _option("potion_courier", "grab_potions")

    assert setting_1.combat_encounter == "battleworn_dummy"
    assert setting_1.random_potion_count == 1
    assert setting_1.metadata["combat_target_hp"] == 75
    assert setting_3.random_relic_count == 1
    assert chest.remove_card_ids == ("lantern_key",)
    assert chest.random_potion_count == 2
    assert chest.random_relic_count == 2
    assert bottle.fixed_potion_ids == ("glowwater_potion",)
    assert climb.fixed_relic_ids == ("fresnel_lens",)
    assert climb.max_hp_delta == -13
    assert tea.fixed_relic_ids == ("royal_poison",)
    assert tea.heal_percent_max_hp == pytest.approx(1.0)
    assert courier.fixed_potion_ids == ("foul_potion", "foul_potion", "foul_potion")


def test_upgrade_transform_and_remove_markers_are_cataloged() -> None:
    aroma_transform = _option("aroma_of_chaos", "let_go")
    aroma_upgrade = _option("aroma_of_chaos", "maintain_control")
    dummy_upgrade = _option("battleworn_dummy", "setting_2")
    emotional = _option("zen_weaver", "emotional_awareness")
    acupuncture = _option("zen_weaver", "arachnid_acupuncture")
    wellspring = _option("wellspring", "bathe")
    bird = _option("wood_carvings", "bird")
    torus = _option("wood_carvings", "torus")

    assert aroma_transform.transform_random_count == 1
    assert aroma_transform.metadata["catalog_status"] == "supported"
    assert aroma_upgrade.upgrade_random_count == 1
    assert dummy_upgrade.upgrade_random_count == 2
    assert emotional.gold_delta == -125
    assert emotional.remove_random_count == 1
    assert acupuncture.gold_delta == -250
    assert acupuncture.remove_random_count == 2
    assert wellspring.remove_random_count == 1
    assert wellspring.fixed_card_ids == ("guilty",)
    assert bird.transform_random_count == 1
    assert bird.metadata["transform_target_card_id"] == "peck"
    assert torus.metadata["transform_target_card_id"] == "toric_toughness"


def test_lantern_key_override_preserves_existing_post_combat_reward_marker() -> None:
    keep = _option("the_lantern_key", "keep_the_key")
    returned = _option("the_lantern_key", "return_the_key")

    assert returned.gold_delta == 100
    assert keep.combat_encounter == "normal"
    assert keep.fixed_card_ids == ("lantern_key",)
    assert keep.metadata["reward_timing"] == "post_combat"
    assert keep.metadata["catalog_status"] == "partial"


def test_coverage_metadata_exposes_unsupported_and_bespoke_categories() -> None:
    wongos = event_catalog_coverage("welcome_to_wongos")
    categories = event_catalog_unsupported_categories()

    assert _coverage("welcome_to_wongos", "mystery_box").status == "partial"
    assert "delayed_combat_effect" in _coverage(
        "welcome_to_wongos",
        "mystery_box",
    ).categories
    assert _coverage("welcome_to_wongos", "leave").status == "unsupported"
    assert "downgrade" in _coverage("welcome_to_wongos", "leave").categories
    assert _coverage("welcome_to_wongos", "bargain_bin_locked").status == "unsupported"
    assert "locked_option" in _coverage(
        "welcome_to_wongos",
        "bargain_bin_locked",
    ).categories
    assert "enchant" in categories
    assert "random_card_add" in categories
    assert "relic_trade" in categories
    assert any(row.status == "supported" for row in wongos)

    with pytest.raises(ValueError, match="Unknown event id"):
        known_event_catalog_options("missing_event")
