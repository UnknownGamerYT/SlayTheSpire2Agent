from __future__ import annotations

import json
from typing import Any

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import (
    CardEnchantment,
    CardInstance,
    CombatState,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    PlayerState,
    RoomKind,
    RunPhase,
    RunState,
)
from sts2sim.engine.rng import new_rng_state


def _card(
    *,
    instance_id: str = "glam:1",
    effects: dict[str, Any] | None = None,
    target: str = "enemy",
    cost: int = 1,
) -> CardInstance:
    return CardInstance(
        instance_id=instance_id,
        card_id="glam_attack",
        type="attack",
        cost=cost,
        target=target,
        effects=effects or {"damage": 6},
        enchantments=(CardEnchantment(keyword="glam", amount=1, source_id="silken_tress"),),
        custom={"star_cost": 1, "x_cost": cost < 0},
    )


def _combat(
    card: CardInstance | None = None,
    *,
    enemy_hp: tuple[int, ...] = (100,),
    metadata: dict[str, Any] | None = None,
    relics: tuple[str, ...] = (),
) -> RunState:
    card = card or _card()
    player = PlayerState(hp=80, max_hp=80, energy=3, resources={"star": 3})
    monsters = tuple(
        MonsterState(monster_id=f"enemy:{index}", hp=hp, max_hp=hp, intent="unknown")
        for index, hp in enumerate(enemy_hp)
    )
    return RunState(
        seed=137,
        character_id="TEST",
        rng=new_rng_state(137),
        player=player,
        master_deck=(card,),
        relics=relics,
        combat=CombatState(
            player=player,
            monsters=monsters,
            hand=(card,),
            draw_per_turn=1,
            metadata=metadata or {},
        ),
        flags={"combat_reward_card_count": 0, "combat_reward_relic_count": 0},
    )


def _play(state: RunState, instance_id: str = "glam:1") -> RunState:
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "play_card" and action.card_instance_id == instance_id
        ),
    )


def _end_turn(state: RunState) -> RunState:
    return step(state, next(action for action in legal_actions(state) if action.type == "end_turn"))


def _glam_replays(state: RunState) -> int:
    return sum(
        event.kind == "card_extra_played" and event.metadata.get("keyword") == "glam"
        for entry in state.replay_log
        for event in entry.events
    )


def test_glam_replays_first_physical_card_play_once_without_extra_costs() -> None:
    state = _play(_combat())
    assert state.combat.monsters[0].hp == 88
    assert state.player.energy == 2
    assert state.player.resources["star"] == 2
    assert len(state.combat.discard_pile) == 1
    assert _glam_replays(state) == 1

    state = _play(_end_turn(state))
    assert state.combat.monsters[0].hp == 82
    assert state.player.energy == 2
    assert state.player.resources["star"] == 1
    assert _glam_replays(state) == 1


def test_glam_consumption_survives_save_reload() -> None:
    state = _end_turn(_play(_combat()))
    restored = load_state(json.loads(json.dumps(serialize(state))))

    assert serialize(_play(state)) == serialize(_play(restored))
    assert _glam_replays(_play(restored)) == 1


def test_glam_resets_on_new_combat() -> None:
    state = _play(_combat(enemy_hp=(12,), relics=("divine_right",)))
    assert state.phase == RunPhase.REWARD
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    fight = MapNodeState(node_id="fight", act=1, floor=1, lane=0, kind=RoomKind.MONSTER)
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "combat": None,
            "reward": None,
            "map": MapState(
                act=1,
                nodes=(start, fight),
                edges=(MapEdgeState(from_id="start", to_id="fight"),),
                current_node_id="start",
                completed_node_ids=("start",),
            ),
        }
    )
    state = step(
        state, next(action for action in legal_actions(state) if action.type == "choose_node")
    )
    starting_hp = state.combat.monsters[0].hp
    state = _play(state)

    assert state.combat.monsters[0].hp == starting_hp - 12
    assert _glam_replays(state) == 2


def test_glam_does_not_retarget_or_replay_other_effects_after_single_target_dies() -> None:
    state = _play(
        _combat(
            _card(effects={"sequence": [{"damage": 6}, {"block": 4}]}),
            enemy_hp=(6, 100),
        )
    )

    assert [monster.hp for monster in state.combat.monsters] == [0, 100]
    assert state.player.block == 4
    assert _glam_replays(state) == 0
    assert any(
        event.kind == "glam_replay_skipped" and event.metadata["reason"] == "target_defeated"
        for event in state.replay_log[-1].events
    )


def test_glam_replays_all_enemy_effects_when_an_enemy_dies() -> None:
    state = _play(
        _combat(
            _card(effects={"all_damage": 6, "block": 4}, target="all_enemies"),
            enemy_hp=(6, 100),
        )
    )

    assert [monster.hp for monster in state.combat.monsters] == [0, 88]
    assert state.player.block == 8
    assert _glam_replays(state) == 1


def test_glam_reuses_paid_x_energy_for_effects_without_charging_again() -> None:
    state = _play(_combat(_card(cost=-1, effects={"damage": {"per_energy": 2}})))

    assert state.player.energy == 0
    assert state.player.resources["star"] == 2
    assert state.combat.monsters[0].hp == 88
    assert _glam_replays(state) == 1


def test_glam_does_not_repeat_itself_through_other_replay_effects() -> None:
    state = _play(
        _combat(
            metadata={
                "next_card_extra_play": ({"card_type": "any", "amount": 1},),
            }
        )
    )

    assert state.combat.monsters[0].hp == 82
    assert _glam_replays(state) == 1


def test_glam_does_not_repeat_itself_through_synthetic_relic_replays() -> None:
    state = _play(_combat(relics=("throwing_axe",)))

    assert state.combat.monsters[0].hp == 82
    assert _glam_replays(state) == 1


def test_glam_charge_belongs_to_each_physical_card_not_its_content_id() -> None:
    first, second = _card(), _card(instance_id="glam:2")
    state = _combat(first)
    state = state.model_copy(
        update={
            "master_deck": (first, second),
            "combat": state.combat.model_copy(update={"hand": (first, second)}),
        }
    )
    state = _play(_play(state), "glam:2")

    assert state.combat.monsters[0].hp == 76
    assert state.player.energy == 1
    assert state.player.resources["star"] == 1
    assert _glam_replays(state) == 2
