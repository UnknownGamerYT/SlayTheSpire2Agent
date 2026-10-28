from __future__ import annotations

import pytest

from sts2sim import action_space, encode_observation, legal_actions, load_state, serialize, step
from sts2sim.engine import (
    Action,
    CardInstance,
    CombatState,
    EventOptionState,
    EventState,
    PlayerState,
    RewardState,
    RunPhase,
    RunState,
    ShopItemState,
    ShopState,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import (
    IllegalActionError,
    _apply_relic_pickup_effects,
    _enchant_card,
)

CHOSEN_RELICS = (
    "archaic_tooth",
    "astrolabe",
    "beautiful_bracelet",
    "biiig_hug",
    "claws",
    "dollys_mirror",
    "electric_shrymp",
    "empty_cage",
    "gnarled_hammer",
    "kifuda",
    "new_leaf",
    "paels_growth",
    "pomander",
    "precarious_shears",
    "precise_scissors",
    "preserved_fog",
    "punch_dagger",
    "royal_stamp",
    "tri_boomerang",
    "yummy_cookie",
)


def _card(index: int, *, upgraded: bool = False, eternal: bool = False) -> CardInstance:
    skill = index % 2 == 1
    return CardInstance(
        instance_id=f"deck:{index}",
        card_id="defend_ironclad" if skill else "bash",
        type="skill" if skill else "attack",
        target="self" if skill else "enemy",
        cost=4,
        effects={"block": 5} if skill else {"damage": 8},
        upgraded=upgraded,
        tags=("eternal",) if eternal else (),
    )


def _reward_state(relic_id: str, *, cards: tuple[CardInstance, ...] | None = None) -> RunState:
    return RunState(
        seed=70,
        character_id="IRONCLAD",
        phase=RunPhase.REWARD,
        rng=new_rng_state(70),
        player=PlayerState(hp=60, max_hp=80, gold=100),
        master_deck=cards if cards is not None else tuple(_card(index) for index in range(8)),
        reward=RewardState(reward_id="test-reward", relic_id=relic_id, gold=77),
    )


def _pickup(relic_id: str, **kwargs) -> RunState:
    state = _reward_state(relic_id, **kwargs)
    action = next(action for action in legal_actions(state) if action.type == "take_reward_relic")
    return step(state, action)


def _choose(state: RunState, instance_id: str | None = None) -> RunState:
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card"
        and (instance_id is None or action.card_instance_id == instance_id)
    )
    return step(state, action)


@pytest.mark.parametrize("relic_id", CHOSEN_RELICS)
def test_every_chosen_pickup_exposes_exclusive_card_actions_and_finishes(relic_id: str) -> None:
    state = _pickup(relic_id)
    assert state.reward is not None and state.reward.relic_claimed
    assert any(action.type == "choose_card" for action in legal_actions(state))
    assert all(action.type in {"choose_card", "proceed"} for action in legal_actions(state))
    for _index in range(8):
        if not state.flags.get("pending_relic_deck_choices"):
            break
        state = _choose(state)
    assert not state.flags.get("pending_relic_deck_choices")
    assert state.phase == RunPhase.REWARD
    assert state.reward is not None and state.reward.gold == 77
    assert any(action.type == "take_reward_gold" for action in legal_actions(state))
    assert not any(action.type == "take_reward_relic" for action in legal_actions(state))


def test_paels_growth_enchants_the_chosen_card_and_is_visible_to_the_agent() -> None:
    state = _pickup("paels_growth")
    observation = encode_observation(state, include_state=False)
    assert observation["relic_deck_choices"][0]["relic_id"] == "paels_growth"
    descriptors = action_space(state)
    assert all(row["card"]["zone"] == "master_deck" for row in descriptors)
    assert all(row["payload"]["choice_kind"] == "modify_deck_cards" for row in descriptors)
    before_rng = state.rng

    result = _choose(state, "deck:3")

    chosen = next(card for card in result.master_deck if card.instance_id == "deck:3")
    assert chosen.custom["enchant_keyword"] == "clone"
    assert chosen.enchantments[0].keyword == "clone"
    assert chosen.enchantments[0].source_id == "paels_growth"
    assert result.rng == before_rng  # Selecting a card does not pick a random substitute.
    assert len([card for card in result.master_deck if card.enchantments]) == 1


def test_biiig_hug_requires_four_distinct_removals_before_other_rewards() -> None:
    state = _pickup("biiig_hug")
    first_action = next(
        action for action in legal_actions(state) if action.card_instance_id == "deck:0"
    )
    assert all(action.type == "choose_card" for action in legal_actions(state))
    with pytest.raises(IllegalActionError):
        step(state, Action(type="take_reward_gold", target_id="reward:gold"))
    state = step(state, first_action)
    with pytest.raises(IllegalActionError):
        step(state, first_action)
    state = load_state(serialize(state))
    for selected in ("deck:1", "deck:2", "deck:3"):
        state = _choose(state, selected)

    assert [card.instance_id for card in state.master_deck] == [
        f"deck:{index}" for index in range(4, 8)
    ]
    assert len(state.flags["relic_removed_card_specs"]) == 4
    assert state.player.gold == 100
    gold = next(action for action in legal_actions(state) if action.type == "take_reward_gold")
    state = step(state, gold)
    assert state.player.gold == 177
    with pytest.raises(IllegalActionError):
        step(state, gold)


@pytest.mark.parametrize("relic_id", ["claws", "gnarled_hammer", "kifuda"])
def test_up_to_choices_can_finish_with_zero_or_fewer_than_the_limit(relic_id: str) -> None:
    state = _pickup(relic_id)
    finish = next(action for action in legal_actions(state) if action.type == "proceed")
    initial_deck = state.master_deck
    skipped = step(state, finish)
    assert skipped.master_deck == initial_deck
    assert not skipped.flags.get("pending_relic_deck_choices")

    state = _choose(state)
    finish = next(action for action in legal_actions(state) if action.type == "proceed")
    result = step(state, finish)
    assert not result.flags.get("pending_relic_deck_choices")
    assert (
        sum(
            card != original
            for card, original in zip(result.master_deck, initial_deck, strict=True)
        )
        == 1
    )


def test_chosen_swift_and_sharp_change_card_effects_and_respect_type_filters() -> None:
    state = _pickup("beautiful_bracelet")
    state = _choose(state, "deck:3")
    assert next(card for card in state.master_deck if card.instance_id == "deck:3").cost == 1

    state = _pickup("gnarled_hammer")
    assert all(
        action.card_instance_id in {"deck:0", "deck:2", "deck:4", "deck:6"}
        for action in legal_actions(state)
        if action.type == "choose_card"
    )
    state = _choose(state, "deck:2")
    assert (
        next(card for card in state.master_deck if card.instance_id == "deck:2").effects["damage"]
        == 11
    )
    assert not any(action.card_instance_id == "deck:2" for action in legal_actions(state))


def test_claws_uses_fixed_maul_transform_and_preserves_selected_instance_id() -> None:
    state = _pickup("claws")
    state = _choose(state, "deck:5")
    card = next(card for card in state.master_deck if card.instance_id == "deck:5")
    assert card.card_id == "maul"
    assert not any(action.card_instance_id == "deck:5" for action in legal_actions(state))


@pytest.mark.parametrize(
    ("starter", "result_id"),
    [
        ("bash", "break"),
        ("neutralize", "suppress"),
        ("falling_star", "meteor_shower"),
        ("unleash", "protector"),
        ("dualcast", "quadcast"),
    ],
)
def test_archaic_tooth_uses_exact_starter_mapping_and_preserves_enchants(
    starter: str, result_id: str
) -> None:
    original = _card(0).model_copy(update={"card_id": starter, "upgraded": True})
    original = _enchant_card(original, "Sharp", 2, source_id="previous-relic")
    state = _pickup("archaic_tooth", cards=(original, _card(1)))
    assert [action.card_instance_id for action in legal_actions(state)] == ["deck:0"]

    state = _choose(state, "deck:0")

    card = state.master_deck[0]
    assert card.card_id == result_id
    assert card.instance_id == original.instance_id
    assert card.upgraded is True
    assert card.enchantments == original.enchantments


def test_archaic_tooth_honors_eggs_without_counting_transformation_as_a_pickup() -> None:
    state = _reward_state("archaic_tooth", cards=(_card(0),))
    state = state.model_copy(update={"relics": ("molten_egg",)})
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))
    state = _choose(state)
    assert state.master_deck[0].card_id == "break"
    assert state.master_deck[0].upgraded is True
    assert not any(event.kind == "reward_card_taken" for event in state.replay_log[-1].events)


def test_selected_transform_replays_identically_and_keeps_astrolabe_upgrades() -> None:
    state = _pickup("astrolabe")
    first = _choose(state, "deck:2")
    restored = load_state(serialize(first))
    second = _choose(first, "deck:4")
    replayed = _choose(restored, "deck:4")
    assert serialize(second) == serialize(replayed)
    final = _choose(second, "deck:0")
    assert not final.flags.get("pending_relic_deck_choices")
    assert all(
        card.upgraded
        for card in final.master_deck
        if card.instance_id in {"deck:0", "deck:2", "deck:4"}
    )


def test_dollys_mirror_copies_only_the_selected_card_with_a_fresh_unique_id() -> None:
    cards = (_card(0).model_copy(update={"instance_id": "bash:3"}), _card(1))
    state = _pickup("dollys_mirror", cards=cards)
    result = _choose(state, "bash:3")
    assert len(result.master_deck) == 3
    assert len({card.instance_id for card in result.master_deck}) == 3
    assert result.master_deck[-1].model_dump(exclude={"instance_id"}) == cards[0].model_dump(
        exclude={"instance_id"}
    )


def test_no_eligible_cards_does_not_deadlock_or_offer_invalid_selections() -> None:
    state = _pickup("electric_shrymp", cards=(_card(0),))
    assert not state.flags.get("pending_relic_deck_choices")
    assert not any(action.type == "choose_card" for action in legal_actions(state))
    assert any(
        event.kind == "relic_deck_choice_unavailable" for event in state.replay_log[-1].events
    )
    state = _pickup("yummy_cookie", cards=(_card(0, upgraded=True), _card(1, eternal=True)))
    assert not state.flags.get("pending_relic_deck_choices")


def test_precarious_shears_does_not_repeat_pickup_hp_loss_for_each_selection() -> None:
    state = _pickup("precarious_shears")
    assert state.player.hp == 44
    state = _choose(_choose(state))
    assert state.player.hp == 44
    assert not state.flags.get("pending_relic_deck_choices")


def test_ancient_numeric_effects_can_already_be_applied_while_choices_still_queue() -> None:
    state = _reward_state("beautiful_bracelet")
    state, events = _apply_relic_pickup_effects(
        state, "beautiful_bracelet", apply_table_effects=False
    )
    assert any(event.kind == "relic_deck_choice_required" for event in events)
    assert len(legal_actions(state)) == 8
    assert state.player.hp == 60
    assert state.player.gold == 100


def test_shop_relic_pickup_queues_once_and_charges_only_the_purchase() -> None:
    state = _reward_state("dollys_mirror")
    state = state.model_copy(
        update={
            "phase": RunPhase.SHOP,
            "reward": None,
            "shop": ShopState(
                node_id="shop",
                items=(
                    ShopItemState(
                        slot_id="0",
                        item_id="dollys_mirror",
                        kind="relic",
                        rarity="rare",
                        price=40,
                        base_price=40,
                    ),
                ),
            ),
        }
    )
    purchase = next(action for action in legal_actions(state) if action.type == "shop_buy")
    state = step(state, purchase)
    assert state.player.gold == 60
    assert all(action.type == "choose_card" for action in legal_actions(state))

    state = _choose(state, "deck:2")

    assert state.player.gold == 60
    assert state.phase == RunPhase.SHOP
    assert any(action.type == "shop_leave" for action in legal_actions(state))
    with pytest.raises(IllegalActionError):
        step(state, purchase)


def test_relic_choice_describes_master_deck_card_instead_of_a_temporary_combat_copy() -> None:
    state = _pickup("paels_growth")
    card = state.master_deck[0]
    state = state.model_copy(
        update={
            "combat": CombatState(player=state.player, hand=(card.model_copy(update={"cost": 0}),))
        }
    )

    descriptor = next(
        row for row in action_space(state) if row["card_instance_id"] == card.instance_id
    )

    assert descriptor["card"]["zone"] == "master_deck"
    assert descriptor["card"]["cost"] == 4
    assert descriptor["relic"]["relic_id"] == "paels_growth"


def test_finishing_optional_relic_choice_is_not_described_as_skipping_rewards() -> None:
    state = _pickup("kifuda")
    finish = next(row for row in action_space(state) if row["type"] == "proceed")

    assert "reward_choice" not in finish
    assert "event_option" not in finish
    result = step(state, finish["action"])
    assert any(action.type == "take_reward_gold" for action in legal_actions(result))


def test_scroll_boxes_pack_descriptors_include_all_real_card_effects_and_identities() -> None:
    pack = ("strike_ironclad", "defend_ironclad", "bash")
    state = _reward_state("scroll_boxes")
    state = state.model_copy(
        update={
            "phase": RunPhase.EVENT,
            "reward": None,
            "event": EventState(
                event_id="scroll_boxes",
                name="Scroll Boxes",
                options=(
                    EventOptionState(
                        option_id="pack:1",
                        title="First pack",
                        metadata={
                            "source_relic_id": "scroll_boxes",
                            "card_ids": pack,
                            "fixed_card_ids": pack,
                        },
                    ),
                ),
            ),
        }
    )

    descriptor = next(row for row in action_space(state) if row["type"] == "choose_event")

    assert len(descriptor["selected_cards"]) == 3
    assert {card["card_id"] for card in descriptor["selected_cards"]} == set(pack)
    assert descriptor["card"]["effect_amounts"]["damage"] >= 14
    assert descriptor["card"]["effect_amounts"]["block"] >= 5
    assert descriptor["mechanics"]["values"]["card_gain"] == 3


def test_new_leaf_chosen_transform_consumes_only_source_niche_rng_and_survives_reload():
    from sts2sim.engine.rng import RunRng, capture_random_state

    state = _reward_state("new_leaf")
    state = state.model_copy(update={"rng": capture_random_state(RunRng(
        "NEW_LEAF", derivation="sts2.v0.107.1.utf16-djb2-splitmix64")),
        "flags": {**state.flags, "cards": (
            {"id": "ANGER", "name": "Anger", "character": "IRONCLAD",
             "rarity": "common", "type": "attack", "damage": 7},
            {"id": "SHRUG_IT_OFF", "name": "Shrug It Off", "character": "IRONCLAD",
             "rarity": "common", "type": "skill", "block": 7},
        )}})
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))
    before = state.rng
    state = _choose(load_state(serialize(state)), "deck:0")
    assert next(c for c in state.master_deck if c.instance_id == "deck:0").card_id in {
        "anger", "shrug_it_off"}
    changed = {key for key, value in state.rng.streams.items()
               if value != before.streams.get(key)}
    assert changed == {"niche"}


def test_neows_talisman_upgrades_last_basic_strike_and_defend_without_rng():
    from sts2sim.engine.transitions import _card_from_spec
    cards = tuple(_card_from_spec({"id": card_id, "instance_id": instance,
        "type": card_type, "rarity": rarity, "cost": 1, "upgraded": upgraded}, i)
        for i, (card_id, instance, card_type, rarity, upgraded) in enumerate((
            ("STRIKE_IRONCLAD", "first", "attack", "basic", False),
            ("STRIKE_IRONCLAD", "last", "attack", "basic", True),
            ("DEFEND_IRONCLAD", "guard", "skill", "basic", False),
            ("PERFECTED_STRIKE", "nonbasic", "attack", "common", False),
        )))
    state = _reward_state("neows_talisman", cards=cards)
    before = state.rng
    state = step(load_state(serialize(state)), next(a for a in legal_actions(state)
                 if a.type == "take_reward_relic"))
    assert [c.upgraded for c in state.master_deck] == [False, True, True, False]
    assert state.rng == before


def test_nutritious_soup_enchants_only_eligible_basic_strikes_with_ember():
    from sts2sim.engine.transitions import _card_from_spec
    cards = tuple(_card_from_spec({"id": card_id, "instance_id": instance,
        "type": "attack", "rarity": rarity, "cost": 1, "damage": 6}, i)
        for i, (card_id, instance, rarity) in enumerate((
            ("STRIKE_IRONCLAD", "basic", "basic"),
            ("PERFECTED_STRIKE", "nonbasic", "common"),
            ("COUNTERSTRIKE", "false_match", "basic"),
        )))
    state = _reward_state("nutritious_soup", cards=cards)
    state = step(load_state(serialize(state)), next(a for a in legal_actions(state)
                 if a.type == "take_reward_relic"))
    assert [len(c.enchantments) for c in state.master_deck] == [1, 0, 0]
    card = state.master_deck[0]
    assert card.enchantments[0].keyword == "tezcataras_ember"
    assert card.enchantments[0].amount == 1
    assert card.cost == 0 and card.custom["eternal"]


def test_paels_claw_uses_defend_tag_and_enchantment_eligibility():
    from sts2sim.engine.transitions import _card_from_spec
    cards = (
        _card_from_spec({"id": "ULTIMATE_DEFEND", "instance_id": "tagged",
                         "type": "skill", "rarity": "ancient", "cost": 1}, 0),
        _card_from_spec({"id": "DEFEND_IRONCLAD", "instance_id": "already",
                         "type": "skill", "cost": 1, "enchantments": (
                             {"keyword": "sharp", "amount": 1},)}, 1),
        _card_from_spec({"id": "DEFENDER", "instance_id": "false_match",
                         "type": "skill", "cost": 1}, 2),
    )
    state = _reward_state("paels_claw", cards=cards)
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))
    assert state.master_deck[0].enchantments[0].keyword == "goopy"
    assert state.master_deck[0].enchantments[0].amount == 1
    assert state.master_deck[0].exhausts
    assert state.master_deck[1].enchantments[0].keyword == "sharp"
    assert not state.master_deck[2].enchantments


def test_paels_growth_chosen_clone_amount_is_four():
    state = _choose(load_state(serialize(_pickup("paels_growth"))), "deck:3")
    card = next(c for c in state.master_deck if c.instance_id == "deck:3")
    assert card.enchantments[0].keyword == "clone"
    assert card.enchantments[0].amount == 4
