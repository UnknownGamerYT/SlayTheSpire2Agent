from __future__ import annotations

from sts2sim.agent_api import action_space, encode_observation
from sts2sim.engine.models import (
    CardInstance,
    CardType,
    CombatState,
    EventOptionState,
    EventState,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    PlayerState,
    RewardState,
    RoomKind,
    RunPhase,
    RunState,
    ShopItemState,
    ShopState,
    TargetType,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.learning.masked_ppo import ACTION_FEATURE_DIM, _action_features


def _state(**updates: object) -> RunState:
    base = RunState(
        seed=1,
        character_id="IRONCLAD",
        ascension=0,
        rng=new_rng_state(1),
        phase=RunPhase.REWARD,
        player=PlayerState(hp=70, max_hp=80, gold=250),
    )
    return base.model_copy(update=updates)


def test_compact_observation_includes_static_identity_buckets() -> None:
    state = _state(
        master_deck=(
            CardInstance(
                instance_id="strike-1",
                card_id="strike",
                type=CardType.ATTACK,
                effects={"damage": 6},
            ),
        ),
        relics=("anchor", "potion_belt"),
        potions=("fire_potion",),
        player=PlayerState(
            hp=70,
            max_hp=80,
            gold=250,
            statuses={"strength": 2},
        ),
    )

    observation = encode_observation(state, include_state=False)

    assert len(observation["vector_schema"]) == len(observation["vector"])
    assert any(name.startswith("owned_relic_") for name in observation["vector_schema"])
    assert any(name.startswith("owned_potion_") for name in observation["vector_schema"])
    assert any(name.startswith("player_status_") for name in observation["vector_schema"])
    assert any(name.startswith("deck_card_") for name in observation["vector_schema"])
    assert any(name.startswith("deck_effect_") for name in observation["vector_schema"])
    assert any(name.startswith("card_position_") for name in observation["vector_schema"])
    assert any(name.startswith("potion_slot_position_") for name in observation["vector_schema"])
    assert "aggression" in observation
    assert "aggression_target" in observation["vector_schema"]
    assert "aggression_hp_floor" in observation["vector_schema"]
    assert "aggression_block_priority" in observation["vector_schema"]
    assert 0.0 <= observation["aggression"]["target"] <= 1.0
    assert "belief" in observation
    assert "reward_plan" in observation
    assert "route_plan" in observation
    assert "visibility" in observation
    assert "player_status_atoms" in observation["visibility"]
    assert "card_slots" in observation["visibility"]
    assert "reward_option_slots" in observation["visibility"]
    assert "shop_option_slots" in observation["visibility"]
    assert any(name.startswith("belief_") for name in observation["vector_schema"])
    assert any(name.startswith("reward_plan_") for name in observation["vector_schema"])
    assert any(name.startswith("route_plan_") for name in observation["vector_schema"])
    assert any(name.startswith("player_status_atom_") for name in observation["vector_schema"])
    assert any(name.startswith("enemy_trait_") for name in observation["vector_schema"])
    assert any(name.startswith("trigger_visibility_") for name in observation["vector_schema"])
    assert any(name.startswith("hand_slot_") for name in observation["vector_schema"])
    assert any(name.startswith("reward_option_slot_") for name in observation["vector_schema"])


def test_combat_play_card_action_exposes_effect_amounts() -> None:
    card = CardInstance(
        instance_id="bash-1",
        card_id="bash",
        type=CardType.ATTACK,
        cost=2,
        target=TargetType.ENEMY,
        effects={"damage": 8, "apply_status": {"status": "vulnerable", "amount": 2}},
    )
    state = _state(
        phase=RunPhase.COMBAT,
        combat=CombatState(
            player=PlayerState(hp=70, max_hp=80, energy=3),
            hand=(card,),
            monsters=(MonsterState(monster_id="jaw_worm", hp=40, max_hp=40),),
        ),
    )

    play = next(action for action in action_space(state) if action["type"] == "play_card")

    assert play["card"]["card_id"] == "bash"
    assert play["card"]["zone"] == "hand"
    assert play["card"]["position"] == 0
    assert play["card"]["effect_amounts"]["damage"] == 8
    assert "apply_status" in play["card"]["effect_keys"]
    assert play["mechanics"]["values"]["damage"] == 8.0
    assert play["mechanics"]["values"]["vulnerable"] == 2.0
    assert "status:vulnerable" in play["mechanics"]["tags"]
    assert play["synergy"]["values"]["frontload"] > 0.0
    assert play["synergy"]["content_id"] == "bash"
    assert play["preview"]["target_hp_delta"] == -8
    assert play["preview"]["player_energy_delta"] == -2
    assert play["preview"]["hand_delta"] == -1
    assert "projected_damage_taken_after_end" in play["preview"]
    assert len(_action_features(play)) == ACTION_FEATURE_DIM


def test_potion_actions_expose_tactical_timing_context() -> None:
    state = _state(
        phase=RunPhase.COMBAT,
        potions=("fire_potion", "block_potion", "attack_potion"),
        combat=CombatState(
            player=PlayerState(hp=10, max_hp=80, energy=3),
            monsters=(
                MonsterState(
                    monster_id="jaw_worm",
                    hp=18,
                    max_hp=40,
                    intent="attack",
                    intent_damage=20,
                ),
            ),
        ),
    )

    actions = action_space(state)
    fire = next(
        action
        for action in actions
        if action["type"] == "use_potion"
        and action["potion"]["potion_id"] == "fire_potion"
    )
    block = next(
        action
        for action in actions
        if action["type"] == "use_potion"
        and action["potion"]["potion_id"] == "block_potion"
    )
    attack = next(
        action
        for action in actions
        if action["type"] == "use_potion"
        and action["potion"]["potion_id"] == "attack_potion"
    )

    assert fire["potion_strategy"]["target_lethal_now"] is True
    assert fire["potion_strategy"]["lethal_now"] is True
    assert fire["potion_strategy"]["overkill_damage"] == 2
    assert "lethal_now" in fire["potion_strategy"]["roles"]
    assert block["potion_strategy"]["survival_enabling"] is True
    assert block["potion_strategy"]["damage_prevented_this_turn"] >= 12
    assert "prevents_death" in block["potion_strategy"]["roles"]
    assert attack["potion_strategy"]["card_generation"] == 3
    assert attack["potion_strategy"]["free_card_play"] == 1
    assert "preemptive_fight_setup" in attack["potion_strategy"]["roles"]
    assert len(_action_features(fire)) == ACTION_FEATURE_DIM
    assert len(_action_features(block)) == ACTION_FEATURE_DIM
    assert len(_action_features(attack)) == ACTION_FEATURE_DIM


def test_foul_potion_merchant_throw_exposes_consumed_slot() -> None:
    state = _state(
        phase=RunPhase.SHOP,
        potions=("fire_potion", "foul_potion"),
        shop=ShopState(node_id="shop-1", items=()),
    )

    throw = next(
        action
        for action in action_space(state)
        if action["type"] == "throw_potion_at_merchant"
    )

    assert throw["payload"]["potion_slot"] == "potion:1"
    assert throw["potion"]["potion_id"] == "foul_potion"
    assert throw["potion"]["slot_index"] == 1
    assert throw["potion_strategy"]["frees_slot"] is True
    assert throw["mechanics"]["values"]["gold_delta"] == 100.0
    assert throw["mechanics"]["values"]["potion_loss"] == -1.0
    assert "merchant_throw" in throw["mechanics"]["tags"]
    assert len(_action_features(throw)) == ACTION_FEATURE_DIM


def test_fake_merchant_foul_potion_throw_exposes_combat_start_not_gold() -> None:
    state = _state(
        phase=RunPhase.EVENT,
        potions=("foul_potion",),
        event=EventState(
            event_id="fake_merchant",
            name="Fake Merchant",
            options=(
                EventOptionState(
                    option_id="SUMMARY",
                    title="Fake Merchant Summary",
                    metadata={"summary_marker": True},
                ),
            ),
        ),
    )

    throw = next(
        action
        for action in action_space(state)
        if action["type"] == "throw_potion_at_merchant"
    )

    assert throw["target_id"] == "fake_merchant"
    assert throw["payload"]["merchant"] == "fake_merchant"
    assert throw["potion"]["potion_id"] == "foul_potion"
    assert throw["potion_strategy"]["aoe_damage"] == 0
    assert throw["potion_strategy"]["self_damage"] == 0
    assert throw["mechanics"]["values"].get("gold_delta", 0.0) == 0.0
    assert throw["mechanics"]["values"]["potion_loss"] == -1.0
    assert "fake_merchant_throw" in throw["mechanics"]["tags"]
    assert "gain_gold" not in throw["mechanics"]["tags"]
    assert len(_action_features(throw)) == ACTION_FEATURE_DIM


def test_enemy_turn_preview_exposes_retaliation_and_next_turn_context() -> None:
    state = _state(
        phase=RunPhase.COMBAT,
        combat=CombatState(
            player=PlayerState(
                hp=70,
                max_hp=80,
                energy=3,
                statuses={"thorns": 3},
            ),
            monsters=(
                MonsterState(
                    monster_id="cultist",
                    hp=3,
                    max_hp=48,
                    intent="attack",
                    intent_damage=6,
                ),
            ),
        ),
    )

    end_turn = next(action for action in action_space(state) if action["type"] == "end_turn")

    assert end_turn["preview"]["projected_damage_taken_after_end"] == 6
    assert end_turn["preview"]["enemy_turn_available"] == 1
    assert end_turn["preview"]["enemy_turn_damage_taken"] == 6
    assert end_turn["preview"]["enemy_turn_retaliation_damage"] == 3
    assert end_turn["preview"]["enemy_turn_retaliation_kills"] == 1
    assert end_turn["preview"]["enemy_turn_monsters_killed"] == 1
    assert end_turn["preview"]["enemy_turn_survives"] == 1
    assert end_turn["preview"]["enemy_turn_death_pending"] == 0
    assert end_turn["preview"]["enemy_turn_next_incoming_damage"] == 0
    assert len(_action_features(end_turn)) == ACTION_FEATURE_DIM


def test_reward_and_shop_actions_expose_choice_identity() -> None:
    reward_state = _state(
        phase=RunPhase.REWARD,
        reward=RewardState(
            reward_id="reward-1",
            source="combat",
            card_options=("pommel_strike", "shrug_it_off"),
            relic_id="anchor",
            potion_id="fire_potion",
        ),
    )
    reward_actions = action_space(reward_state)
    reward_card = next(
        action
        for action in reward_actions
        if action["type"] == "take_reward_card"
        and action["target_id"] == "reward:card:0"
    )
    reward_skip_card_group = next(
        action
        for action in reward_actions
        if action["type"] == "skip_reward"
        and action["target_id"] == "reward:card_options"
    )
    reward_proceed = next(action for action in reward_actions if action["type"] == "proceed")
    reward_relic = next(
        action for action in reward_actions if action["type"] == "take_reward_relic"
    )
    reward_potion = next(
        action for action in reward_actions if action["type"] == "take_reward_potion"
    )

    assert reward_card["card"]["card_id"] == "pommel_strike"
    assert reward_card["reward_choice"]["content_id"] == "pommel_strike"
    assert reward_card["reward_choice"]["selection_set_id"] == "card_options"
    assert reward_card["reward_choice"]["selection_set_size"] == 2
    assert reward_card["reward_choice"]["sibling_content_ids"] == ["shrug_it_off"]
    assert reward_card["reward_bundle"]["can_skip"] is True
    assert reward_card["reward_bundle"]["available_counts"]["total"] == 4
    assert reward_card["reward_bundle"]["available_counts"]["selection_sets"] == 3
    assert "gain_card" in reward_card["mechanics"]["tags"]
    assert reward_card["synergy"]["values"]["frontload"] > 0.0
    assert "deck:adds_card" in reward_card["synergy"]["tags"]
    assert reward_card["option_slot"]["content_id"] == "pommel_strike"
    assert reward_card["option_slot"]["values"]["card_gain"] == 1.0
    assert reward_relic["relic"]["relic_id"] == "anchor"
    assert reward_relic["mechanics"]["values"]["relic_gain"] == 1.0
    assert reward_relic["mechanics"]["values"]["block"] >= 10.0
    assert reward_relic["synergy"]["values"]["relic_synergy"] >= 0.0
    assert "timing:start_of_combat" in reward_relic["mechanics"]["tags"]
    assert reward_potion["potion"]["potion_id"] == "fire_potion"
    assert reward_potion["mechanics"]["values"]["damage"] >= 20.0
    assert reward_skip_card_group["reward_choice"]["kind"] == "skip"
    assert reward_skip_card_group["reward_choice"]["skip_kind"] == "card_options"
    assert reward_skip_card_group["reward_choice"]["skips_selection"] is True
    assert reward_skip_card_group["reward_choice"]["skips_remaining"] is False
    assert reward_skip_card_group["reward_choice"]["selection_set_id"] == "card_options"
    assert reward_skip_card_group["reward_choice"]["selection_set_size"] == 2
    assert reward_skip_card_group["reward_choice"]["sibling_content_ids"] == [
        "pommel_strike",
        "shrug_it_off",
    ]
    assert reward_skip_card_group["option_slot"]["kind"] == "skip"
    assert reward_skip_card_group["option_slot"]["content_id"] == "card_options"
    assert reward_skip_card_group["option_slot"]["skip_action"] is True
    assert reward_proceed["reward_choice"]["kind"] == "proceed"
    assert reward_proceed["reward_choice"]["skips_remaining"] is True
    assert reward_proceed["reward_choice"]["available_remaining_count"] == 4
    assert reward_proceed["option_slot"]["kind"] == "proceed"
    assert reward_proceed["option_slot"]["skip_action"] is True
    assert len(_action_features(reward_card)) == ACTION_FEATURE_DIM
    assert len(_action_features(reward_skip_card_group)) == ACTION_FEATURE_DIM
    assert len(_action_features(reward_proceed)) == ACTION_FEATURE_DIM

    shop_state = _state(
        phase=RunPhase.SHOP,
        shop=ShopState(
            node_id="shop-1",
            items=(
                ShopItemState(
                    slot_id="shop:0",
                    item_id="anchor",
                    kind="relic",
                    rarity="common",
                    price=175,
                    base_price=175,
                ),
            ),
        ),
    )
    shop_buy = next(action for action in action_space(shop_state) if action["type"] == "shop_buy")

    assert shop_buy["item"]["item_id"] == "anchor"
    assert shop_buy["item"]["price"] == 175
    assert shop_buy["relic"]["relic_id"] == "anchor"
    assert shop_buy["mechanics"]["values"]["gold_delta"] == -175.0
    assert shop_buy["mechanics"]["values"]["block"] >= 10.0
    assert "gain_relic" in shop_buy["mechanics"]["tags"]
    assert shop_buy["synergy"]["values"]["gold_pressure"] > 0.0
    assert shop_buy["option_slot"]["content_id"] == "anchor"
    assert shop_buy["option_slot"]["values"]["gold_pressure"] > 0.0
    assert len(_action_features(shop_buy)) == ACTION_FEATURE_DIM


def test_event_multi_card_choice_exposes_selected_cards() -> None:
    strike = CardInstance(
        instance_id="strike-1",
        card_id="strike",
        type=CardType.ATTACK,
        effects={"damage": 6},
    )
    defend = CardInstance(
        instance_id="defend-1",
        card_id="defend",
        type=CardType.SKILL,
        target=TargetType.SELF,
        effects={"block": 5},
    )
    state = _state(
        phase=RunPhase.EVENT,
        master_deck=(strike, defend),
        event=EventState(
            event_id="SELF_HELP_BOOK",
            name="Self Help Book",
            options=(
                EventOptionState(
                    option_id="ENCHANT_TWO",
                    title="Enchant two",
                    metadata={
                        "enchant": True,
                        "enchant_count": 2,
                        "enchant_keyword": "Sharp",
                        "enchant_amount": 2,
                    },
                ),
            ),
        ),
    )

    event_action = next(
        action for action in action_space(state) if action["type"] == "choose_event"
    )

    assert event_action["selected_card_count"] == 2
    assert [card["instance_id"] for card in event_action["selected_cards"]] == [
        "strike-1",
        "defend-1",
    ]
    assert event_action["selected_cards"][0]["zone"] == "master_deck"
    assert event_action["selected_cards"][1]["position"] == 1
    assert event_action["card"]["selected_count"] == 2
    assert event_action["card"]["effect_amounts"]["damage"] == 6.0
    assert event_action["card"]["effect_amounts"]["block"] == 5.0
    assert event_action["event_option"]["option_id"] == "ENCHANT_TWO"
    assert "event_option" in event_action["mechanics"]["tags"]
    assert len(_action_features(event_action)) == ACTION_FEATURE_DIM


def test_event_options_expose_marker_payload_and_skip_semantics() -> None:
    state = _state(
        phase=RunPhase.EVENT,
        event=EventState(
            event_id="TRIAL",
            name="Trial",
            page_id="NONDESCRIPT",
            options=(
                EventOptionState(
                    option_id="GUILTY",
                    title="DECIDE: Guilty",
                    description="Add Doubt. Gain 2 card rewards.",
                    metadata={
                        "fixed_card_ids": ("doubt",),
                        "card_reward_count": 2,
                    },
                ),
            ),
        ),
    )

    actions = action_space(state)
    guilty = next(action for action in actions if action["type"] == "choose_event")
    skip = next(action for action in actions if action["type"] == "proceed")

    assert guilty["event_option"]["metadata"]["fixed_card_ids"] == ["doubt"]
    assert guilty["event_option"]["metadata"]["card_reward_count"] == 2
    assert guilty["mechanics"]["values"]["card_gain"] == 3.0
    assert "gain_card" in guilty["mechanics"]["tags"]
    assert skip["event_option"]["option_id"] == "__skip_event__"
    assert skip["event_option"]["skip_action"] is True
    assert skip["event_option"]["available_option_ids"] == ["GUILTY"]
    assert "skip_event" in skip["mechanics"]["tags"]
    assert len(_action_features(guilty)) == ACTION_FEATURE_DIM
    assert len(_action_features(skip)) == ACTION_FEATURE_DIM


def test_event_transform_options_split_chosen_and_random_semantics() -> None:
    strike = CardInstance(
        instance_id="strike-1",
        card_id="strike",
        type=CardType.ATTACK,
        effects={"damage": 6},
    )
    chosen_state = _state(
        phase=RunPhase.EVENT,
        master_deck=(strike,),
        event=EventState(
            event_id="TRANSFORMER",
            name="Transformer",
            options=(
                EventOptionState(
                    option_id="CHOSEN",
                    title="Transform chosen",
                    metadata={"transform_card_count": 1},
                ),
            ),
        ),
    )
    random_state = _state(
        phase=RunPhase.EVENT,
        master_deck=(strike,),
        event=EventState(
            event_id="TRANSFORMER",
            name="Transformer",
            options=(
                EventOptionState(
                    option_id="RANDOM",
                    title="Transform random",
                    metadata={
                        "transform_card_count": 2,
                        "transform_random_card_count": 2,
                    },
                ),
            ),
        ),
    )

    chosen = next(
        action for action in action_space(chosen_state) if action["type"] == "choose_event"
    )
    random = next(
        action for action in action_space(random_state) if action["type"] == "choose_event"
    )

    assert chosen["selected_cards"][0]["instance_id"] == "strike-1"
    assert chosen["mechanics"]["values"]["card_transform"] == 1.0
    assert "chosen_transform" in chosen["mechanics"]["tags"]
    assert "random_transform" not in chosen["mechanics"]["tags"]
    assert "selected_cards" not in random
    assert random["mechanics"]["values"]["card_transform"] == 2.0
    assert random["mechanics"]["values"]["randomness"] == 1.0
    assert "random_transform" in random["mechanics"]["tags"]
    assert "chosen_transform" not in random["mechanics"]["tags"]


def test_optional_reward_remove_exposes_exact_deck_card_and_remove_semantics() -> None:
    strike = CardInstance(
        instance_id="strike-1",
        card_id="strike",
        type=CardType.ATTACK,
        effects={"damage": 6},
    )
    defend = CardInstance(
        instance_id="defend-1",
        card_id="defend",
        type=CardType.SKILL,
        target=TargetType.SELF,
        effects={"block": 5},
    )
    state = _state(
        phase=RunPhase.REWARD,
        master_deck=(strike, defend),
        reward=RewardState(
            reward_id="reward-optional-remove",
            source="combat",
            metadata={
                "optional_remove_card_count": 1,
                "optional_remove_card_instance_ids": ("strike-1", "defend-1"),
                "optional_remove_card_ids": ("strike", "defend"),
            },
        ),
    )

    remove_strike = next(
        action
        for action in action_space(state)
        if action["type"] == "take_reward_card"
        and action["target_id"] == "reward:remove_card:0"
    )

    assert remove_strike["card"]["instance_id"] == "strike-1"
    assert remove_strike["card"]["zone"] == "master_deck"
    assert remove_strike["card"]["position"] == 0
    assert remove_strike["card"]["reward_remove"] is True
    assert remove_strike["reward_choice"]["kind"] == "card_removal"
    assert remove_strike["reward_choice"]["card_instance_id"] == "strike-1"
    assert remove_strike["reward_choice"]["selection_set_id"] == "card_removal"
    assert remove_strike["reward_choice"]["selection_set_size"] == 2
    assert remove_strike["mechanics"]["values"]["card_remove"] == 1.0
    assert remove_strike["mechanics"]["values"].get("card_gain", 0.0) == 0.0
    assert "remove_card" in remove_strike["mechanics"]["tags"]
    assert remove_strike["option_slot"]["kind"] == "card_removal"
    assert remove_strike["option_slot"]["values"]["card_remove"] == 1.0
    assert len(_action_features(remove_strike)) == ACTION_FEATURE_DIM


def test_positions_are_visible_to_observation_and_actions() -> None:
    strike = CardInstance(
        instance_id="strike-1",
        card_id="strike",
        type=CardType.ATTACK,
        effects={"damage": 6},
    )
    defend = CardInstance(
        instance_id="defend-1",
        card_id="defend",
        type=CardType.SKILL,
        target=TargetType.SELF,
        effects={"block": 5},
    )
    bash = CardInstance(
        instance_id="bash-1",
        card_id="bash",
        type=CardType.ATTACK,
        effects={"damage": 8},
    )
    shrug = CardInstance(
        instance_id="shrug-1",
        card_id="shrug_it_off",
        type=CardType.SKILL,
        effects={"block": 8, "draw": 1},
    )
    state = _state(
        phase=RunPhase.COMBAT,
        master_deck=(strike, defend, bash, shrug),
        potions=("fire_potion", "block_potion"),
        combat=CombatState(
            player=PlayerState(hp=70, max_hp=80, energy=3),
            hand=(strike, defend),
            draw_pile=(bash, shrug),
            monsters=(MonsterState(monster_id="jaw_worm", hp=40, max_hp=40),),
        ),
    )

    observation = encode_observation(state, include_state=False)
    defend_action = next(
        action
        for action in action_space(state)
        if action["type"] == "play_card" and action["card"]["card_id"] == "defend"
    )

    assert observation["positions"]["cards"]["hand"][1]["card_id"] == "defend"
    assert observation["positions"]["cards"]["draw_pile"][0]["card_id"] == "bash"
    assert observation["positions"]["cards"]["draw_pile"][0]["position_from_top"] == 0
    assert observation["positions"]["potions"][1]["potion_id"] == "block_potion"
    assert defend_action["card"]["zone"] == "hand"
    assert defend_action["card"]["position"] == 1
    assert len(_action_features(defend_action)) == ACTION_FEATURE_DIM


def test_timed_effects_are_visible_to_observation_and_actions() -> None:
    setup = CardInstance(
        instance_id="setup-1",
        card_id="setup",
        type=CardType.SKILL,
        target=TargetType.SELF,
        effects={
            "combat_trigger": {
                "trigger": "turn_start",
                "duration": "once",
                "delay": 2,
                "remaining_uses": 1,
                "effects": ({"kind": "draw", "amount": 1, "target": "player"},),
            }
        },
    )
    state = _state(
        phase=RunPhase.COMBAT,
        master_deck=(setup,),
        player=PlayerState(
            hp=70,
            max_hp=80,
            gold=250,
            statuses={"next_turn_energy": 2},
        ),
        combat=CombatState(
            turn=3,
            player=PlayerState(hp=70, max_hp=80, energy=3),
            hand=(setup,),
            monsters=(MonsterState(monster_id="jaw_worm", hp=40, max_hp=40),),
            metadata={
                "timed_card_triggers": (
                    {
                        "source_card_id": "setup",
                        "trigger": "turn_start",
                        "duration": "once",
                        "delay": 2,
                        "remaining_uses": 1,
                        "effects": ({"kind": "draw", "amount": 1, "target": "player"},),
                    },
                )
            },
        ),
    )

    observation = encode_observation(state, include_state=False)
    play = next(action for action in action_space(state) if action["type"] == "play_card")

    assert play["mechanics"]["values"]["turn_delay"] >= 2.0
    assert play["mechanics"]["values"]["turns_until_effect"] >= 2.0
    assert play["mechanics"]["values"]["remaining_uses"] >= 1.0
    assert "timing:turn_start" in play["mechanics"]["tags"]
    assert observation["mechanics"]["values"]["turn_delay"] >= 2.0
    assert observation["mechanics"]["values"]["next_turn_effect"] >= 1.0
    assert observation["mechanics"]["values"]["energy"] >= 2.0
    assert "active_trigger" in observation["mechanics"]["tags"]


def test_target_context_distinguishes_multi_enemy_actions() -> None:
    card = CardInstance(
        instance_id="strike-1",
        card_id="strike",
        type=CardType.ATTACK,
        target=TargetType.ENEMY,
        effects={"damage": 6},
    )
    state = _state(
        phase=RunPhase.COMBAT,
        combat=CombatState(
            player=PlayerState(hp=70, max_hp=80, energy=3),
            hand=(card,),
            monsters=(
                MonsterState(
                    monster_id="cultist_a",
                    hp=7,
                    max_hp=48,
                    block=0,
                    intent="attack",
                    intent_damage=6,
                    statuses={"vulnerable": 2},
                    metadata={"source_monster_id": "cultist", "slot_index": 0},
                ),
                MonsterState(
                    monster_id="jaw_worm_b",
                    hp=40,
                    max_hp=40,
                    block=12,
                    intent="attack_defend",
                    intent_damage=11,
                    statuses={"strength": 3, "weak": 1},
                    metadata={"source_monster_id": "jaw_worm", "slot_index": 1},
                ),
            ),
        ),
    )

    observation = encode_observation(state, include_state=False)
    plays = [action for action in action_space(state) if action["type"] == "play_card"]
    by_target = {action["target"]["target_id"]: action for action in plays}

    assert observation["targets"]["monsters"][0]["statuses"]["vulnerable"] == 2
    assert observation["targets"]["monsters"][1]["block"] == 12
    assert observation["mechanics"]["values"]["vulnerable"] >= 2.0
    assert observation["mechanics"]["values"]["weak"] >= 1.0
    assert by_target["cultist_a"]["target"]["hp"] == 7
    assert by_target["jaw_worm_b"]["target"]["block"] == 12
    assert by_target["jaw_worm_b"]["target"]["statuses"]["strength"] == 3
    assert _action_features(by_target["cultist_a"]) != _action_features(by_target["jaw_worm_b"])


def test_map_choice_actions_include_path_lookahead() -> None:
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    elite = MapNodeState(node_id="elite", act=1, floor=1, lane=0, kind=RoomKind.ELITE)
    shop = MapNodeState(node_id="shop", act=1, floor=1, lane=1, kind=RoomKind.SHOP)
    rest = MapNodeState(node_id="rest", act=1, floor=2, lane=0, kind=RoomKind.REST)
    monster = MapNodeState(node_id="monster", act=1, floor=2, lane=1, kind=RoomKind.MONSTER)
    boss = MapNodeState(node_id="boss", act=1, floor=3, lane=0, kind=RoomKind.BOSS)
    game_map = MapState(
        act=1,
        nodes=(start, elite, shop, rest, monster, boss),
        edges=(
            MapEdgeState(from_id="start", to_id="elite"),
            MapEdgeState(from_id="start", to_id="shop"),
            MapEdgeState(from_id="elite", to_id="rest"),
            MapEdgeState(from_id="shop", to_id="monster"),
            MapEdgeState(from_id="rest", to_id="boss"),
            MapEdgeState(from_id="monster", to_id="boss"),
        ),
        current_node_id="start",
        completed_node_ids=("start",),
        boss_node_id="boss",
    )
    state = _state(
        phase=RunPhase.MAP,
        floor=0,
        map=game_map,
        master_deck=(
            CardInstance(
                instance_id="bash-1",
                card_id="bash",
                type=CardType.ATTACK,
                effects={"damage": 8},
            ),
        ),
    )

    choices = [action for action in action_space(state) if action["type"] == "choose_node"]
    by_kind = {action["node"]["kind"]: action for action in choices}

    assert by_kind["elite"]["node"]["path"]["max_elites"] == 1
    assert by_kind["elite"]["node"]["path"]["max_rests"] == 1
    assert by_kind["shop"]["node"]["path"]["max_shops"] == 1
    assert by_kind["shop"]["node"]["path"]["max_monsters"] == 1
    assert by_kind["elite"]["node"]["path"]["has_boss_path"] is True
    assert by_kind["elite"]["node"]["path"]["future_relic_rewards_max"] == 2
    assert by_kind["shop"]["node"]["path"]["future_relic_rewards_max"] == 1
    assert by_kind["elite"]["node"]["path"]["first_rest_depth_min"] == 2
    assert by_kind["shop"]["node"]["path"]["paths_with_rest_fraction"] == 0
    assert by_kind["elite"]["node"]["path"]["current_hp_fraction"] == 0.875
    assert by_kind["elite"]["node"]["path"]["upgradeable_card_count"] == 1
    assert (
        by_kind["elite"]["node"]["path"]["avg_aggression_score"]
        > by_kind["shop"]["node"]["path"]["avg_aggression_score"]
    )
    assert _action_features(by_kind["elite"]) != _action_features(by_kind["shop"])


def test_observation_and_actions_carry_shared_mechanic_context() -> None:
    state = _state(
        phase=RunPhase.REWARD,
        master_deck=(
            CardInstance(
                instance_id="defend-1",
                card_id="defend",
                type=CardType.SKILL,
                target=TargetType.SELF,
                effects={"block": 5},
            ),
        ),
        relics=("anchor",),
        potions=("fire_potion",),
        reward=RewardState(
            reward_id="reward-1",
            source="combat",
            potion_id="fire_potion",
        ),
    )

    observation = encode_observation(state, include_state=False)
    reward_potion = next(
        action for action in action_space(state) if action["type"] == "take_reward_potion"
    )

    assert "mechanics" in observation
    assert observation["mechanics"]["values"]["block"] >= 15.0
    assert observation["mechanics"]["values"]["damage"] >= 20.0
    assert reward_potion["mechanics"]["values"]["potion_gain"] == 1.0
    assert reward_potion["mechanics"]["values"]["damage"] >= 20.0
