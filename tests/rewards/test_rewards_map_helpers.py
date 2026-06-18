from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fakes import (
    LocalGameRng,
    generate_card_rewards,
    generate_map,
    reachable_next_nodes,
)
from helpers import call_with_supported_kwargs, canonical_digest, import_attr, jsonable


def _reward_generator() -> Any:
    return import_attr(
        ("sts2sim.mechanics.rewards", "sts2sim.api.rewards"),
        ("generate_card_rewards", "card_rewards", "roll_card_rewards"),
    ) or generate_card_rewards


def _map_generator() -> Any:
    return import_attr(
        ("sts2sim.mechanics.map", "sts2sim.api.map"),
        ("generate_map", "build_act_map", "create_map"),
    ) or generate_map


def _reachable_helper() -> Any:
    return import_attr(
        ("sts2sim.mechanics.map", "sts2sim.api.map"),
        ("reachable_next_nodes", "next_nodes", "reachable_from"),
    ) or reachable_next_nodes


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get("id") or f"{node['floor']}-{node['lane']}")


def test_card_rewards_are_unique_and_seed_stable() -> None:
    pool = [
        {"id": "strike_plus", "rarity": "common"},
        {"id": "defend_plus", "rarity": "common"},
        {"id": "bash_plus", "rarity": "uncommon"},
        {"id": "limit_break", "rarity": "rare"},
    ]

    first = call_with_supported_kwargs(
        _reward_generator(),
        pool=pool,
        card_pool=pool,
        rng=LocalGameRng(7),
        count=3,
    )
    second = call_with_supported_kwargs(
        _reward_generator(),
        pool=pool,
        card_pool=pool,
        rng=LocalGameRng(7),
        count=3,
    )

    first_payload = jsonable(first)
    ids = [card["id"] for card in first_payload]

    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert canonical_digest(first_payload) == canonical_digest(second)


def test_generated_map_has_next_floor_edges() -> None:
    game_map = call_with_supported_kwargs(
        _map_generator(),
        rng=LocalGameRng(99),
        floors=5,
        width=3,
    )
    payload = jsonable(game_map)
    nodes = payload["nodes"]
    nodes_by_id = {_node_id(node): node for node in nodes}
    reachable = _reachable_helper()

    assert len(nodes) == 15
    assert payload["start_ids"] == ["0-0", "0-1", "0-2"]

    for node in nodes:
        node_id = _node_id(node)
        next_ids = call_with_supported_kwargs(
            reachable,
            game_map=payload,
            map=payload,
            node=node,
            node_id=node_id,
        )
        if node["floor"] == 4:
            assert list(next_ids) == []
            continue

        assert next_ids
        for next_id in next_ids:
            assert nodes_by_id[str(next_id)]["floor"] == node["floor"] + 1
