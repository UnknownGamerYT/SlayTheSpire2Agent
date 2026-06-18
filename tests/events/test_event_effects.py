from __future__ import annotations

import json
from typing import Any

from helpers import project_root

from sts2sim.content.event_effects import (
    EventTextEffect,
    EventTextEffectKind,
    parse_event_option_effects,
)

EVENTS_PATH = project_root() / "data" / "cache" / "eng" / "events.json"


def _events() -> list[dict[str, Any]]:
    data = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def _option_description(event_name: str, option_id: str) -> str:
    for event in _events():
        if event.get("name") != event_name:
            continue
        for option in event.get("options") or ():
            if option.get("id") == option_id:
                description = option.get("description")
                assert isinstance(description, str)
                return description
    raise AssertionError(f"Missing cached event option: {event_name} / {option_id}")


def _assert_description_in_cache(description: str) -> None:
    for event in _events():
        option_sets = [event.get("options") or ()]
        option_sets.extend((page.get("options") or ()) for page in event.get("pages") or ())
        for options in option_sets:
            if any(option.get("description") == description for option in options):
                return
    raise AssertionError(f"Description is not present in cached events: {description}")


def _amount_tuple(
    effect: EventTextEffect,
) -> tuple[int | None, int | None, int | None, int | None, bool]:
    amount = effect.amount
    if amount is None:
        return (None, None, None, None, False)
    return (amount.value, amount.minimum, amount.maximum, amount.percent, amount.full)


def test_battleworn_dummy_options_parse_fights_rewards_and_upgrades() -> None:
    setting_1 = parse_event_option_effects(_option_description("Battleworn Dummy", "SETTING_1"))
    assert [effect.kind for effect in setting_1] == [
        EventTextEffectKind.FIGHT,
        EventTextEffectKind.POTION_PROCURE,
    ]
    assert _amount_tuple(setting_1[0]) == (75, None, None, None, False)
    assert setting_1[0].target == "dummy"
    assert setting_1[1].count == 1
    assert setting_1[1].random is True

    setting_2 = parse_event_option_effects(_option_description("Battleworn Dummy", "SETTING_2"))
    assert [effect.kind for effect in setting_2] == [
        EventTextEffectKind.FIGHT,
        EventTextEffectKind.CARD_UPGRADE,
    ]
    assert _amount_tuple(setting_2[0]) == (150, None, None, None, False)
    assert setting_2[1].count == 2
    assert setting_2[1].random is True

    setting_3 = parse_event_option_effects(_option_description("Battleworn Dummy", "SETTING_3"))
    assert [effect.kind for effect in setting_3] == [
        EventTextEffectKind.FIGHT,
        EventTextEffectKind.RELIC_OBTAIN,
    ]
    assert _amount_tuple(setting_3[0]) == (300, None, None, None, False)
    assert setting_3[1].count == 1
    assert setting_3[1].random is True


def test_war_historian_repy_options_parse_key_cost_and_rewards() -> None:
    cage = parse_event_option_effects(_option_description("War Historian, Repy", "UNLOCK_CAGE"))
    assert [effect.kind for effect in cage] == [
        EventTextEffectKind.CARD_REMOVE,
        EventTextEffectKind.RELIC_OBTAIN,
    ]
    assert cage[0].item_name == "Lantern Key"
    assert cage[0].count == 1
    assert cage[1].item_name == "History Course"
    assert cage[1].random is False

    chest = parse_event_option_effects(_option_description("War Historian, Repy", "UNLOCK_CHEST"))
    assert [effect.kind for effect in chest] == [
        EventTextEffectKind.CARD_REMOVE,
        EventTextEffectKind.POTION_PROCURE,
        EventTextEffectKind.RELIC_OBTAIN,
    ]
    assert chest[0].item_name == "Lantern Key"
    assert chest[1].count == 2
    assert chest[1].random is True
    assert chest[2].count == 2
    assert chest[2].random is True


def test_lantern_key_options_parse_gold_and_fight_marker() -> None:
    returned = parse_event_option_effects(_option_description("The Lantern Key", "RETURN_THE_KEY"))
    assert [effect.kind for effect in returned] == [EventTextEffectKind.GOLD_GAIN]
    assert _amount_tuple(returned[0]) == (100, None, None, None, False)

    kept = parse_event_option_effects(_option_description("The Lantern Key", "KEEP_THE_KEY"))
    assert [effect.kind for effect in kept] == [EventTextEffectKind.FIGHT]
    assert kept[0].qualifier == "to obtain the Key"


def test_round_tea_party_options_parse_fixed_relic_full_heal_and_hp_cost() -> None:
    tea = parse_event_option_effects(_option_description("The Round Tea Party", "ENJOY_TEA"))
    assert [effect.kind for effect in tea] == [
        EventTextEffectKind.RELIC_OBTAIN,
        EventTextEffectKind.HP_HEAL,
    ]
    assert tea[0].item_name == "Royal Poison"
    assert _amount_tuple(tea[1]) == (None, None, None, None, True)

    fight = parse_event_option_effects(_option_description("The Round Tea Party", "PICK_FIGHT"))
    assert [effect.kind for effect in fight] == [
        EventTextEffectKind.HP_LOSS,
        EventTextEffectKind.RELIC_OBTAIN,
    ]
    assert _amount_tuple(fight[0]) == (11, None, None, None, False)
    assert fight[1].random is True


def test_punch_off_options_parse_curse_relic_and_fight() -> None:
    nab = parse_event_option_effects(_option_description("Punch Off", "NAB"))
    assert [effect.kind for effect in nab] == [
        EventTextEffectKind.CARD_ADD,
        EventTextEffectKind.RELIC_OBTAIN,
    ]
    assert nab[0].item_name == "Injury"
    assert nab[0].target == "deck"
    assert nab[1].random is True

    fight = parse_event_option_effects(_option_description("Punch Off", "I_CAN_TAKE_THEM"))
    assert [effect.kind for effect in fight] == [EventTextEffectKind.FIGHT]
    assert fight[0].target == "them"
    assert fight[0].qualifier == "Greater Rewards"


def test_dense_vegetation_options_parse_gold_hp_percent_heal_and_fight() -> None:
    trudge = parse_event_option_effects(_option_description("Dense Vegetation", "TRUDGE_ON"))
    assert [effect.kind for effect in trudge] == [
        EventTextEffectKind.GOLD_GAIN,
        EventTextEffectKind.HP_LOSS,
    ]
    assert _amount_tuple(trudge[0]) == (None, 61, 99, None, False)
    assert _amount_tuple(trudge[1]) == (8, None, None, None, False)

    rest = parse_event_option_effects(_option_description("Dense Vegetation", "REST"))
    assert [effect.kind for effect in rest] == [
        EventTextEffectKind.HP_HEAL,
        EventTextEffectKind.FIGHT,
    ]
    assert _amount_tuple(rest[0]) == (None, None, None, 30, False)
    assert rest[1].target == "enemies"


def test_parser_handles_other_cached_primitives_requested_for_audit() -> None:
    examples = {
        "Gain [green]6[/green] Max HP.": (
            EventTextEffectKind.MAX_HP_GAIN,
            (6, None, None, None, False),
        ),
        (
            "Lose [red]26-44[/red] [gold]Gold[/gold]. "
            "Procure [blue]2[/blue] random [gold]Potions[/gold]."
        ): (
            EventTextEffectKind.GOLD_LOSS,
            (None, 26, 44, None, False),
        ),
        (
            "Remove [blue]2[/blue] [gold]Strikes[/gold]. "
            "Add [gold]Ultimate Strike[/gold] to your [gold]Deck[/gold]."
        ): (
            EventTextEffectKind.CARD_REMOVE,
            (None, None, None, None, False),
        ),
        "Choose [blue]1[/blue] starter card to [gold]Transform[/gold] into [gold]Peck[/gold].": (
            EventTextEffectKind.CARD_TRANSFORM,
            (None, None, None, None, False),
        ),
    }
    for description, (first_kind, first_amount) in examples.items():
        _assert_description_in_cache(description)
        parsed = parse_event_option_effects(description)
        assert parsed
        assert parsed[0].kind is first_kind
        assert _amount_tuple(parsed[0]) == first_amount

    removed_and_added = parse_event_option_effects(
        "Remove [blue]2[/blue] [gold]Strikes[/gold]. "
        "Add [gold]Ultimate Strike[/gold] to your [gold]Deck[/gold]."
    )
    assert removed_and_added[0].item_name == "Strike"
    assert removed_and_added[0].count == 2
    assert removed_and_added[1].item_name == "Ultimate Strike"

    transformed = parse_event_option_effects(
        "Choose [blue]1[/blue] starter card to [gold]Transform[/gold] into [gold]Peck[/gold]."
    )
    assert transformed[0].count == 1
    assert transformed[0].qualifier == "starter"
    assert transformed[0].target == "Peck"
