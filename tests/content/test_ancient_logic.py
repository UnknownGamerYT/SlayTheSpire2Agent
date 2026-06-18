from __future__ import annotations

from typing import Any, TypeVar

from sts2sim.mechanics.ancients import (
    ANCIENT_POSITIVE_RELIC_PAIRS,
    ANCIENT_POSITIVE_RELICS,
    AncientChoice,
    AncientContext,
    AncientMarkerKind,
    ancient_ids_for_act,
    generate_ancient_choices,
    resolve_ancient_choice,
    with_generated_ancient_choices,
)

T = TypeVar("T")


class FirstChoiceRng:
    def choice(self, seq: list[T] | tuple[T, ...]) -> T:
        return seq[0]

    def shuffle(self, x: list[Any]) -> None:
        return None

    def random(self) -> float:
        return 0.0


def test_ancient_option_generation_is_seed_deterministic() -> None:
    context = AncientContext(
        act=1,
        hp=20,
        max_hp=80,
        gold=99,
        deck=("Strike", "Defend"),
    )

    first = generate_ancient_choices(context, seed=1234)
    second = generate_ancient_choices(context, seed=1234)

    assert first == second
    assert len(first) == 3
    assert [choice.option_id for choice in first] == [
        "a1:ancient:1",
        "a1:ancient:2",
        "a1:ancient:3",
    ]
    assert [choice.kind for choice in first].count("positive_relic") == 2
    assert [choice.kind for choice in first].count("curse_relic") == 1
    assert all(choice.fixed_relic_ids == (choice.relic_id,) for choice in first)


def test_ancient_resolution_reports_state_deltas_and_effect_markers() -> None:
    choice = AncientChoice(
        option_id="full_blessing",
        name="Full Blessing",
        kind="positive",
        gold_delta=20,
        hp_delta=-6,
        max_hp_delta=5,
        fixed_card_ids=("Ancient Strike",),
        remove_card_ids=("Strike",),
        fixed_relic_ids=("Golden Pearl",),
        random_relic_count=1,
        fixed_potion_ids=("Fire Potion",),
        random_potion_count=1,
        card_reward_count=1,
        card_reward_size=3,
        card_reward_kind="rare",
        upgrade_random_count=2,
        transform_random_count=1,
        remove_random_count=1,
        set_flags={"ancient_seen": True},
    )
    context = AncientContext(
        act=1,
        hp=30,
        max_hp=40,
        gold=10,
        deck=("Strike", "Defend"),
        choices=(choice,),
    )

    resolution = resolve_ancient_choice(
        context,
        "full_blessing",
        rng=FirstChoiceRng(),
        relic_pool=("Anchor", "Golden Pearl"),
        potion_pool=("Skill Potion",),
    )

    assert resolution.gold_delta == 20
    assert resolution.hp_delta == -6
    assert resolution.max_hp_delta == 5
    assert resolution.added_card_ids == ("ancient_strike",)
    assert resolution.removed_card_ids == ("strike",)
    assert resolution.relic_ids == ("golden_pearl", "anchor")
    assert resolution.potion_ids == ("fire_potion", "skill_potion")
    assert resolution.card_reward_count == 1
    assert resolution.upgrade_random_count == 2
    assert resolution.transform_random_count == 1
    assert resolution.remove_random_count == 1
    assert resolution.flags_set == {"ancient_seen": True}

    assert resolution.state.gold == 30
    assert resolution.state.hp == 24
    assert resolution.state.max_hp == 45
    assert resolution.state.deck == ("defend", "ancient_strike")
    assert resolution.state.relics == ("golden_pearl", "anchor")
    assert resolution.state.potions == ("fire_potion", "skill_potion")
    assert resolution.state.flags["ancient_seen"] is True
    assert resolution.state.chosen_option_ids == ("full_blessing",)

    marker_kinds = {marker.kind for marker in resolution.markers}
    assert AncientMarkerKind.GOLD.value in marker_kinds
    assert AncientMarkerKind.HP.value in marker_kinds
    assert AncientMarkerKind.MAX_HP.value in marker_kinds
    assert AncientMarkerKind.CARD_ADD.value in marker_kinds
    assert AncientMarkerKind.CARD_REMOVE.value in marker_kinds
    assert AncientMarkerKind.FIXED_RELIC.value in marker_kinds
    assert AncientMarkerKind.RANDOM_RELIC.value in marker_kinds
    assert AncientMarkerKind.FIXED_POTION.value in marker_kinds
    assert AncientMarkerKind.RANDOM_POTION.value in marker_kinds
    assert AncientMarkerKind.CARD_REWARD.value in marker_kinds
    assert AncientMarkerKind.CARD_UPGRADE_RANDOM.value in marker_kinds
    assert AncientMarkerKind.CARD_TRANSFORM_RANDOM.value in marker_kinds
    assert AncientMarkerKind.CARD_REMOVE_RANDOM.value in marker_kinds
    assert AncientMarkerKind.FLAG_SET.value in marker_kinds
    assert next(
        marker
        for marker in resolution.markers
        if marker.kind == AncientMarkerKind.RANDOM_RELIC.value
    ).metadata == {"resolved_ids": ("anchor",)}


def test_act_specific_ancient_ids_and_act_two_special_options() -> None:
    assert ancient_ids_for_act(1) == ("neow",)
    assert ancient_ids_for_act(2) == ("orobas", "pael", "tezcatara")
    assert ancient_ids_for_act(3) == ("nonupeipe", "tanx", "vakuu")

    owned_non_act_two_positive_relics = (
        ANCIENT_POSITIVE_RELICS
        + tuple(relic for pair in ANCIENT_POSITIVE_RELIC_PAIRS for relic in pair)
    )
    context = AncientContext(
        act=2,
        ancient_id="orobas",
        relics=owned_non_act_two_positive_relics,
    )

    generated = with_generated_ancient_choices(context, rng=FirstChoiceRng())
    positive_relic_ids = {
        choice.relic_id
        for choice in generated.choices
        if choice.kind == "positive_relic"
    }

    assert generated.ancient_id == "orobas"
    assert positive_relic_ids == {"golden_compass", "prismatic_gem"}
    assert all(choice.metadata["ancient_id"] == "orobas" for choice in generated.choices)
