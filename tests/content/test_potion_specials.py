from __future__ import annotations

from sts2sim.mechanics.potion_specials import (
    PotionEffectClassification,
    classify_potion_effect,
    potion_special_coverage,
)


def _row(potion_id: str, description: str, name: str | None = None) -> dict[str, str]:
    return {
        "id": potion_id,
        "name": name or potion_id.replace("_", " ").title(),
        "description": description,
    }


def _categories(classification: PotionEffectClassification) -> tuple[str, ...]:
    return tuple(blocker.category for blocker in classification.blockers)


def test_classifies_draw_energy_and_timed_turn_partials() -> None:
    swift = classify_potion_effect(_row("SWIFT_POTION", "Draw [blue]3[/blue] cards."))
    cure_all = classify_potion_effect(
        _row("CURE_ALL", "Gain [energy:1]. Draw [blue]2[/blue] cards.")
    )
    clarity = classify_potion_effect(
        _row(
            "CLARITY",
            "Draw [blue]1[/blue] card. At the start of your next [blue]3[/blue] "
            "turns, draw [blue]1[/blue] additional card.",
        )
    )

    assert swift.status == "executable"
    assert swift.executable_steps == ({"draw": 3},)
    assert cure_all.executable_steps == ({"energy": 1}, {"draw": 2})
    assert clarity.status == "partial"
    assert clarity.executable_steps == ({"draw": 1},)
    assert _categories(clarity) == ("timed_turn_start_effect",)


def test_classifies_strength_dexterity_temporary_and_enemy_status_potions() -> None:
    strength = classify_potion_effect(
        _row("STRENGTH_POTION", "Gain [blue]2[/blue] [gold]Strength[/gold].")
    )
    dexterity = classify_potion_effect(
        _row("DEXTERITY_POTION", "Gain [blue]2[/blue] [gold]Dexterity[/gold].")
    )
    flex = classify_potion_effect(
        _row(
            "FLEX_POTION",
            "Gain [blue]5[/blue] [gold]Strength[/gold]. At the end of your turn, "
            "lose [blue]5[/blue] [gold]Strength[/gold].",
        )
    )
    speed = classify_potion_effect(
        _row(
            "SPEED_POTION",
            "Gain [blue]5[/blue] [gold]Dexterity[/gold]. At the end of your turn, "
            "lose [blue]5[/blue] [gold]Dexterity[/gold].",
        )
    )
    binding = classify_potion_effect(
        _row(
            "POTION_OF_BINDING",
            "Apply [blue]1[/blue] [gold]Weak[/gold] and [blue]1[/blue] "
            "[gold]Vulnerable[/gold] to ALL enemies.",
        )
    )

    assert strength.executable_steps == (
        {"apply_status": {"target": "self", "strength": 2}},
    )
    assert dexterity.executable_steps == (
        {"apply_status": {"target": "self", "dexterity": 2}},
    )
    assert flex.executable_steps == (
        {"apply_status": {"target": "self", "strength": 5, "strength_down": 5}},
    )
    assert speed.executable_steps == (
        {"apply_status": {"target": "self", "dexterity": 5, "dexterity_down": 5}},
    )
    assert binding.executable_steps == (
        {"apply_status": {"target": "all_enemies", "weak": 1, "vulnerable": 1}},
    )


def test_classifies_block_plated_armor_and_artifact_like_markers() -> None:
    block = classify_potion_effect(
        _row("BLOCK_POTION", "Gain [blue]12[/blue] [gold]Block[/gold].")
    )
    ship = classify_potion_effect(
        _row(
            "SHIP_IN_A_BOTTLE",
            "Gain [blue]10[/blue] [gold]Block[/gold]. Next turn, gain "
            "[blue]10[/blue] [gold]Block[/gold].",
        )
    )
    plating = classify_potion_effect(
        _row("HEART_OF_IRON", "Gain [blue]7[/blue] [gold]Plating[/gold].")
    )
    artifact = classify_potion_effect(
        _row("ARTIFACT_POTION", "Gain [blue]1[/blue] [gold]Artifact[/gold].")
    )
    buffer = classify_potion_effect(
        _row("LUCKY_TONIC", "Gain [blue]1[/blue] [gold]Buffer[/gold].")
    )

    assert block.executable_steps == ({"block": 12},)
    assert ship.executable_steps == ({"block": 10}, {"next_turn": {"block": 10}})
    assert plating.executable_steps == (
        {"apply_status": {"target": "self", "plated_armor": 7}},
    )
    assert artifact.executable_steps == (
        {"apply_status": {"target": "self", "artifact": 1}},
    )
    assert buffer.executable_steps == (
        {"apply_status": {"target": "self", "buffer": 1}},
    )


def test_classifies_orb_slot_and_channel_potions_as_executable() -> None:
    capacity = classify_potion_effect(
        _row("POTION_OF_CAPACITY", "Gain [blue]2[/blue] [gold]Orb Slots[/gold].")
    )
    darkness = classify_potion_effect(
        _row(
            "ESSENCE_OF_DARKNESS",
            "[gold]Channel[/gold] a [gold]Dark[/gold] for each of your [gold]Orb Slots[/gold].",
        )
    )

    assert capacity.status == "executable"
    assert capacity.executable_steps == ({"orb_slot_delta": 2},)
    assert darkness.status == "executable"
    assert darkness.executable_steps == (
        {"channel_orb": {"orb": "dark", "amount": "orb_slots"}},
    )
    assert darkness.blockers == ()


def test_classifies_random_card_duplication_generation_and_escape_blockers() -> None:
    attack = classify_potion_effect(
        _row(
            "ATTACK_POTION",
            "Choose [blue]1[/blue] of [blue]3[/blue] random Attack cards to add "
            "into your [gold]Hand[/gold]. It's free to play this turn.",
        )
    )
    duplicator = classify_potion_effect(
        _row("DUPLICATOR", "This turn, your next card is played an extra time.")
    )
    entropic = classify_potion_effect(
        _row("ENTROPIC_BREW", "Fill all your empty potion slots with random potions.")
    )
    smoke = classify_potion_effect(
        _row("SMOKE_BOMB", "Escape combat.", name="Smoke Bomb")
    )

    assert attack.status == "blocked"
    assert attack.executable_steps == ()
    assert _categories(attack) == ("random_card_choice",)
    assert _categories(duplicator) == ("card_duplication",)
    assert _categories(entropic) == ("potion_generation",)
    assert _categories(smoke) == ("escape_combat",)


def test_potion_special_coverage_summarizes_steps_and_blockers() -> None:
    summary = potion_special_coverage(
        (
            _row("ENERGY_POTION", "Gain [energy:2]."),
            _row("POTION_OF_CAPACITY", "Gain [blue]2[/blue] [gold]Orb Slots[/gold]."),
            _row("SNECKO_OIL", "Draw [blue]7[/blue] cards. Randomize costs."),
            _row(
                "ATTACK_POTION",
                "Choose [blue]1[/blue] of [blue]3[/blue] random Attack cards.",
            ),
            _row("QUIET_BREW", "Do something mysterious."),
        )
    )

    assert summary.total_rows == 5
    assert summary.executable_rows == 2
    assert summary.partial_rows == 1
    assert summary.blocked_rows == 1
    assert summary.unsupported_rows == 1
    assert summary.covered_rows == 4
    assert summary.executable_effect_keys == ("draw", "energy", "orb_slot_delta")
    assert summary.blocker_categories == (
        "cost_randomization",
        "random_card_choice",
        "unparsed_text",
    )
