from __future__ import annotations

import json
from random import Random

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import (
    AncientOptionState,
    AncientState,
    CardInstance,
    MapEdgeState,
    MapNodeState,
    MapState,
    PlayerState,
    RewardState,
    RoomKind,
    RunPhase,
    RunState,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import (
    _apply_relic_pickup_effects,
    _potion_capacity,
    _shop_item_pool,
    _treasure_relic_pool,
)
from sts2sim.mechanics.relic_lifecycle import ANCIENT_STARTER_RELICS, scroll_boxes_packs
from sts2sim.mechanics.reward_pools import build_character_card_pool
from sts2sim.mechanics.shops import ShopItem, ShopItemKind

PACK_CARDS = tuple(
    {
        "id": f"CARD_{index}",
        "rarity": "Common" if index < 4 else "Uncommon",
        "type": "Skill",
        "color": "test",
        "cost": 1,
        "effects": {"block": index + 1},
    }
    for index in range(6)
)
WAX_RELICS = tuple(
    {"id": name.upper(), "rarity_key": "Common", "pool": "shared"}
    for name in ("anchor", "strawberry", "orichalcum", "bag_of_preparation")
)


def _run(relic: str, *, pool: tuple[dict[str, object], ...] = WAX_RELICS) -> RunState:
    return RunState(
        seed=173,
        character_id="TEST",
        rng=new_rng_state(173),
        phase=RunPhase.TREASURE,
        player=PlayerState(hp=80, max_hp=80, gold=77),
        master_deck=(
            CardInstance(
                instance_id="kill",
                card_id="kill",
                type="attack",
                cost=0,
                target="all_enemies",
                effects={"all_damage": 999},
            ),
        ),
        reward=RewardState(reward_id="fixture", source="treasure", gold=10, relic_id=relic),
        flags={
            "cards": PACK_CARDS,
            "relic_pool": pool,
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 0,
            "combat_reward_gold": 10,
            "combat_reward_potion_chance_percent": 0,
        },
    )


def _action(state: RunState, kind: str, target: str | None = None):
    return next(
        a
        for a in legal_actions(state)
        if a.type == kind and (target is None or a.target_id == target)
    )


def _pickup(state: RunState) -> RunState:
    return step(state, _action(state, "take_reward_relic", "reward:relic"))


def _fight(state: RunState, kind: RoomKind = RoomKind.MONSTER, *, win: bool = True) -> RunState:
    start = MapNodeState(
        node_id="start", act=state.act, floor=state.floor, lane=0, kind=RoomKind.START
    )
    fight = MapNodeState(
        node_id="fight",
        act=state.act,
        floor=state.floor + 1,
        lane=0,
        kind=RoomKind.MONSTER if kind == RoomKind.EVENT else kind,
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "reward": None,
            "event": None,
            "combat": None,
            "map": MapState(
                act=state.act,
                nodes=(start, fight),
                edges=(MapEdgeState(from_id="start", to_id="fight"),),
                current_node_id="start",
                completed_node_ids=("start",),
            ),
        }
    )
    state = step(state, _action(state, "choose_node", "fight"))
    if kind == RoomKind.EVENT:
        state = state.model_copy(
            update={
                "map": state.map.model_copy(
                    update={
                        "nodes": (
                            start,
                            fight.model_copy(update={"kind": RoomKind.EVENT}),
                        )
                    }
                )
            }
        )
    if not win:
        return state
    state = step(state, _action(state, "play_card"))
    assert state.phase == RunPhase.REWARD
    return state


def test_scroll_boxes_offers_forced_distinct_packs_and_adds_exactly_one() -> None:
    state = _pickup(_run("scroll_boxes"))
    assert state.phase == RunPhase.EVENT
    assert state.player.gold == 77
    assert {a.type.value for a in legal_actions(state)} == {"choose_event"}
    packs = state.event.metadata["card_packs"]
    assert len(packs) == 2
    assert len(set(sum(packs, ()))) == 6
    for pack in packs:
        assert sum(card_id in {"card_4", "card_5"} for card_id in pack) == 1
    selected = state.event.options[1]
    state = step(state, _action(state, "choose_event", selected.option_id))
    assert [card.card_id.lower() for card in state.master_deck[1:]] == list(
        selected.metadata["card_ids"]
    )
    assert state.phase == RunPhase.TREASURE
    assert state.reward.relic_claimed is True
    assert state.reward.gold == 10
    assert not state.flags.get("scroll_boxes_pending")
    assert all(a.type.value != "choose_event" for a in legal_actions(state))
    state, events = _apply_relic_pickup_effects(state, "scroll_boxes")
    assert not events
    assert len(state.master_deck) == 4


def test_scroll_boxes_choice_and_rng_survive_reload() -> None:
    state = _pickup(_run("scroll_boxes"))
    restored = load_state(json.loads(json.dumps(serialize(state))))
    assert serialize(step(state, _action(state, "choose_event"))) == serialize(
        step(restored, _action(restored, "choose_event"))
    )


def test_scroll_boxes_ancient_offer_uses_packs_not_skill_reward_approximation() -> None:
    state = new_run(seed=9, character_id="IRONCLAD", ascension=0)
    state = state.model_copy(
        update={
            "ancient": AncientState(
                act=1,
                options=(
                    AncientOptionState(
                        option_id="pack",
                        name="Scroll Boxes",
                        kind="positive_relic",
                        relic_id="scroll_boxes",
                    ),
                ),
            )
        }
    )
    before_rng = state.rng
    state = step(state, _action(state, "choose_ancient", "pack"))
    assert state.phase == RunPhase.EVENT
    assert state.reward is None
    assert state.rng != before_rng
    state = step(state, _action(state, "choose_event"))
    assert state.phase == RunPhase.MAP
    assert len(state.master_deck) == 13


def test_scroll_boxes_missing_source_pool_is_explicitly_blocked() -> None:
    state = _run("scroll_boxes")
    state = state.model_copy(update={"flags": {**state.flags, "cards": PACK_CARDS[:1]}})
    state = _pickup(state)
    assert len(state.master_deck) == 1
    assert "scroll_boxes" in state.flags["unresolved_relic_effects"]
    assert any(e.kind == "relic_effect_blocked" for e in state.replay_log[-1].events)


@pytest.mark.parametrize(
    "rolls,expected", [((0.0, 0.5), 1), ((0.5, 0.0), 1), ((0.0, 0.0), 2), ((0.5, 0.5), 0)]
)
def test_defect_claw_pack_exception_is_independent_for_each_pack(rolls, expected: int) -> None:
    class PackRng(Random):
        def sample(self, population, k, **kwargs):
            return list(population)[:k]

        def random(self):
            return next(self.rolls)

    rng = PackRng(1)
    rng.rolls = iter(rolls)
    cards = build_character_card_pool(
        (
            *({**card, "color": "defect"} for card in PACK_CARDS),
            {"id": "CLAW", "rarity": "Common", "color": "defect", "type": "Attack"},
        ),
        character_id="DEFECT",
    )
    packs = scroll_boxes_packs(rng, cards, character_id="DEFECT")
    assert sum(pack == ("claw",) * 3 for pack in packs) == expected


@pytest.mark.parametrize("starter,upgraded", ANCIENT_STARTER_RELICS.items())
def test_touch_of_orobas_replaces_only_the_owned_starter(starter: str, upgraded: str) -> None:
    state = _run("touch_of_orobas").model_copy(update={"relics": (starter, "anchor")})
    state = _pickup(state)
    assert state.relics == (upgraded, "anchor", "touch_of_orobas")
    restored = load_state(serialize(state))
    unchanged, events = _apply_relic_pickup_effects(restored, "touch_of_orobas")
    assert serialize(unchanged) == serialize(restored)
    assert not events


def test_touch_of_orobas_missing_starter_is_explicitly_blocked() -> None:
    state = _pickup(_run("touch_of_orobas"))
    assert state.relics == ("touch_of_orobas",)
    assert state.flags["unresolved_relic_effects"]["touch_of_orobas"] == "starter_relic_missing"


def _claim_wax(state: RunState) -> RunState:
    for name in ("anchor", "strawberry", "orichalcum", "bag_of_preparation"):
        index = state.reward.relic_ids.index(name)
        state = step(state, _action(state, "take_reward_relic", f"reward:relic:{index}"))
    return state


def test_toy_box_selectable_pickup_order_controls_melting_and_pickup_effects_persist() -> None:
    state = _pickup(_run("toy_box"))
    assert len(state.reward.relic_ids) == 4
    assert state.relics == ("toy_box",)
    assert state.player.max_hp == 80
    state = _claim_wax(state)
    assert state.player.max_hp == 87
    assert state.flags["toy_box_wax_relic_ids"] == (
        "anchor",
        "strawberry",
        "orichalcum",
        "bag_of_preparation",
    )
    for victories in range(1, 13):
        state = _fight(state)
        assert state.flags["relic_counters"]["toy_box"] == victories % 3
        assert len(state.flags["toy_box_wax_relic_ids"]) == 4 - victories // 3
        if victories >= 3:
            assert "anchor" not in state.relics
        if victories >= 6:
            assert "strawberry" not in state.relics
        assert state.player.max_hp == 87
    assert state.flags["toy_box_used_up"] is True
    assert not _treasure_relic_pool(state)
    assert not _shop_item_pool(state, "shop_relic_pool", (ShopItem("anchor", ShopItemKind.RELIC),))


def test_toy_box_decay_survives_reload_and_does_not_repeat_for_rewards() -> None:
    state = _claim_wax(_pickup(_run("toy_box")))
    state = _fight(_fight(state))
    restored = load_state(json.loads(json.dumps(serialize(state))))
    assert serialize(_fight(state)) == serialize(_fight(restored))
    state = _fight(restored)
    wax = state.flags["toy_box_wax_relic_ids"]
    state = step(state, _action(state, "take_reward_gold"))
    assert state.flags["toy_box_wax_relic_ids"] == wax
    assert state.flags["relic_counters"]["toy_box"] == 0


def test_wax_potion_belt_slots_are_applied_once_and_remain_after_melting() -> None:
    pool = (*WAX_RELICS[:3], {"id": "POTION_BELT", "rarity_key": "Common", "pool": "shared"})
    state = _pickup(_run("toy_box", pool=pool))
    index = state.reward.relic_ids.index("potion_belt")
    state = step(state, _action(state, "take_reward_relic", f"reward:relic:{index}"))
    assert _potion_capacity(state) == 5
    for _ in range(3):
        state = _fight(state)
    assert "potion_belt" not in state.relics
    assert _potion_capacity(state) == 5


def test_removing_wax_relic_does_not_reset_toy_box_counter() -> None:
    state = _fight(_claim_wax(_pickup(_run("toy_box"))))
    state = state.model_copy(update={"relics": tuple(r for r in state.relics if r != "anchor")})
    state = _fight(_fight(state))
    assert "strawberry" not in state.relics
    assert state.flags["relic_counters"]["toy_box"] == 0


@pytest.mark.parametrize("kind", [RoomKind.ELITE, RoomKind.BOSS, RoomKind.EVENT])
def test_toy_box_counts_all_completed_encounter_kinds(kind: RoomKind) -> None:
    state = _claim_wax(_pickup(_run("toy_box")))
    state = _fight(_fight(_fight(state), kind), kind)
    assert "anchor" not in state.relics
    assert state.flags["relic_counters"]["toy_box"] == 0


def test_defeat_does_not_tick_toy_box_or_wongo_ticket() -> None:
    state = _claim_wax(_pickup(_run("toy_box")))
    state = state.model_copy(update={"relics": (*state.relics, "wongos_mystery_ticket")})
    state, _ = _apply_relic_pickup_effects(state, "wongos_mystery_ticket")
    state = _fight(state, win=False)
    player = state.combat.player.model_copy(update={"hp": 0})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = step(state, _action(state, "proceed"))
    assert state.phase == RunPhase.FAILED
    assert state.flags["relic_counters"]["toy_box"] == 0
    assert state.flags["relic_counters"]["wongos_mystery_ticket"] == 5
    assert state.flags["delayed_event_rewards"][0]["remaining_combats"] == 5


def test_wongo_ticket_pays_only_after_five_wins_and_claims_apply_pickup_effects() -> None:
    pool = tuple(
        {"id": name, "rarity_key": "Common", "pool": "shared"}
        for name in ("STRAWBERRY", "PEAR", "MANGO")
    )
    state = _pickup(_run("wongos_mystery_ticket", pool=pool))
    for wins in range(1, 6):
        state = _fight(state)
        assert state.flags["relic_counters"]["wongos_mystery_ticket"] == 5 - wins
        assert len(state.reward.relic_ids) == (3 if wins == 5 else 0)
        if wins == 3:
            state = load_state(json.loads(json.dumps(serialize(state))))
    assert not state.flags.get("delayed_event_rewards")
    assert state.player.max_hp == 80
    for index in range(3):
        state = step(state, _action(state, "take_reward_relic", f"reward:relic:{index}"))
    assert state.player.max_hp == 111
    state = _fight(state)
    assert state.reward.relic_ids == ()
    assert state.player.max_hp == 111
    assert state.flags["relic_counters"]["wongos_mystery_ticket"] == 0
