"""Planning summaries for reward screens and map routes.

These helpers operate on serialized state payloads so learning code can consume
context without coupling to engine model instances.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

REWARD_PLAN_KEYS: tuple[str, ...] = (
    "reward_open",
    "forced",
    "can_skip",
    "can_take_multiple_items",
    "available_total",
    "available_selection_sets",
    "available_cards",
    "available_card_groups",
    "available_fixed_cards",
    "available_card_removals",
    "available_relics",
    "available_potions",
    "available_gold",
    "claimed_total",
    "claimed_gold",
    "claimed_primary_card_groups",
    "claimed_card_groups",
    "claimed_fixed_cards",
    "claimed_card_removals",
    "claimed_relics",
    "claimed_potions",
    "skipped_total",
    "skipped_gold",
    "skipped_primary_card_groups",
    "skipped_card_groups",
    "skipped_fixed_cards",
    "skipped_card_removals",
    "skipped_relics",
    "skipped_potions",
)

ROUTE_PLAN_KEYS: tuple[str, ...] = (
    "route_open",
    "reachable_path_count",
    "reachable_next_node_count",
    "boss_path_count",
    "boss_path_fraction",
    "next_act_map_change_pending",
    "golden_compass_act2_map",
    "spoils_map_pending",
    "spoils_map_target_act",
    "spoils_map_target_known",
    "aggressive_score",
    "aggressive_elites",
    "aggressive_rests",
    "aggressive_shops",
    "aggressive_fights",
    "safe_score",
    "safe_elites",
    "safe_rests",
    "safe_shops",
    "safe_fights",
    "upgrade_heavy_score",
    "upgrade_heavy_elites",
    "upgrade_heavy_rests",
    "upgrade_heavy_shops",
    "upgrade_heavy_fights",
    "shop_heavy_score",
    "shop_heavy_elites",
    "shop_heavy_rests",
    "shop_heavy_shops",
    "shop_heavy_fights",
    "elite_heavy_score",
    "elite_heavy_elites",
    "elite_heavy_rests",
    "elite_heavy_shops",
    "elite_heavy_fights",
)

_ROUTE_STYLES: tuple[str, ...] = (
    "aggressive",
    "safe",
    "upgrade_heavy",
    "shop_heavy",
    "elite_heavy",
)


def reward_plan_summary(state_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize all still-available reward choices in one deterministic bundle."""

    reward = _mapping(state_payload.get("reward"))
    if not reward:
        return {
            "reward_open": False,
            "reward_id": "",
            "source": "",
            "forced": False,
            "can_skip": False,
            "can_take_multiple_items": False,
            "available_counts": _empty_reward_counts(),
            "claimed_counts": _empty_claim_skip_counts(),
            "skipped_counts": _empty_claim_skip_counts(),
            "selection_sets": [],
            "available_choices": [],
            "claimed_groups": _reward_group_state(),
            "skipped_groups": _reward_group_state(),
            "available_content_ids": [],
        }

    choices = _available_reward_choices(reward)
    selection_sets = _selection_sets(choices)
    available_counts = _reward_choice_counts(choices)
    claimed_counts = _reward_claimed_counts(reward)
    skipped_counts = _reward_skipped_counts(reward)
    forced = bool(reward.get("forced", False))
    selectable_set_count = sum(1 for item in selection_sets if item["available_count"] > 0)
    return {
        "reward_open": True,
        "reward_id": str(reward.get("reward_id", "")),
        "source": str(reward.get("source", "")),
        "forced": forced,
        "can_skip": not forced,
        "can_take_multiple_items": selectable_set_count > 1,
        "available_counts": available_counts,
        "claimed_counts": claimed_counts,
        "skipped_counts": skipped_counts,
        "selection_sets": selection_sets,
        "available_choices": choices,
        "claimed_groups": _reward_group_state(
            primary_card=bool(reward.get("card_claimed", False)),
            card_groups=_int_sequence(reward.get("claimed_card_option_group_indices")),
            fixed_cards=_int_sequence(reward.get("claimed_card_indices")),
            relics=_str_sequence(reward.get("claimed_relic_ids")),
            potions=_int_sequence(reward.get("claimed_potion_indices")),
        ),
        "skipped_groups": _reward_group_state(
            primary_card=bool(reward.get("card_skipped", False)),
            card_groups=_int_sequence(reward.get("skipped_card_option_group_indices")),
            fixed_cards=_int_sequence(reward.get("skipped_card_indices")),
            relics=_str_sequence(reward.get("skipped_relic_ids")),
            potions=_int_sequence(reward.get("skipped_potion_indices")),
        ),
        "available_content_ids": [
            str(choice["content_id"]) for choice in choices if choice.get("content_id")
        ],
    }


def route_plan_summary(state_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize reachable map paths and choose representative paths by route style."""

    game_map = _mapping(state_payload.get("map"))
    node_by_id = _map_nodes_by_id(game_map)
    current_node_id = str(game_map.get("current_node_id") or "")
    start_ids = _next_route_start_ids(game_map, node_by_id, current_node_id)
    map_effects = _map_effect_summary(state_payload)
    if not game_map or not node_by_id or not start_ids:
        return {
            "route_open": False,
            "current_node_id": current_node_id,
            "reachable_next_node_ids": [],
            "reachable_path_count": 0,
            "boss_path_count": 0,
            "boss_path_fraction": 0.0,
            "map_effects": map_effects,
            "styles": {style: _empty_style_summary(style) for style in _ROUTE_STYLES},
        }

    paths: list[list[str]] = []
    for start_id in start_ids:
        paths.extend(_enumerate_paths(game_map, node_by_id, start_id))
    path_summaries = [_path_summary(path, node_by_id) for path in paths]
    boss_path_count = sum(1 for path in path_summaries if path["ends_at_boss"])
    styles = {
        style: _best_style_path(style, path_summaries) for style in _ROUTE_STYLES
    }
    return {
        "route_open": True,
        "current_node_id": current_node_id,
        "reachable_next_node_ids": start_ids,
        "reachable_path_count": len(path_summaries),
        "boss_path_count": boss_path_count,
        "boss_path_fraction": boss_path_count / max(1, len(path_summaries)),
        "map_effects": map_effects,
        "styles": styles,
    }


def _map_effect_summary(state_payload: Mapping[str, Any]) -> dict[str, Any]:
    flags = _mapping(state_payload.get("flags"))
    game_map = _mapping(state_payload.get("map"))
    current_act = _int(game_map.get("act")) or _int(state_payload.get("act"))
    pending_act = _int(flags.get("spoils_map_pending_act"))
    target_act = _int(flags.get("spoils_map_target_act"))
    golden_compass = bool(flags.get("golden_compass_act2_map"))
    spoils_pending = bool(pending_act or target_act)
    next_act_map_change = bool(
        (golden_compass and current_act < 2)
        or (pending_act and current_act and pending_act > current_act)
        or (target_act and current_act and target_act > current_act)
    )
    return {
        "next_act_map_change_pending": next_act_map_change,
        "golden_compass_act2_map": golden_compass,
        "spoils_map_pending": spoils_pending,
        "spoils_map_target_act": target_act or pending_act,
        "spoils_map_target_known": bool(flags.get("spoils_map_target_node_id")),
    }


def reward_plan_vector(state_payload: Mapping[str, Any]) -> tuple[float, ...]:
    """Return a stable numeric reward-planning vector ordered by REWARD_PLAN_KEYS."""

    summary = reward_plan_summary(state_payload)
    available = _mapping(summary.get("available_counts"))
    claimed = _mapping(summary.get("claimed_counts"))
    skipped = _mapping(summary.get("skipped_counts"))
    values = {
        "reward_open": summary.get("reward_open"),
        "forced": summary.get("forced"),
        "can_skip": summary.get("can_skip"),
        "can_take_multiple_items": summary.get("can_take_multiple_items"),
        "available_total": available.get("total"),
        "available_selection_sets": available.get("selection_sets"),
        "available_cards": available.get("cards"),
        "available_card_groups": available.get("card_groups"),
        "available_fixed_cards": available.get("fixed_cards"),
        "available_card_removals": available.get("card_removals"),
        "available_relics": available.get("relics"),
        "available_potions": available.get("potions"),
        "available_gold": available.get("gold"),
        "claimed_total": claimed.get("total"),
        "claimed_gold": claimed.get("gold"),
        "claimed_primary_card_groups": claimed.get("primary_card_group"),
        "claimed_card_groups": claimed.get("card_groups"),
        "claimed_fixed_cards": claimed.get("fixed_cards"),
        "claimed_card_removals": claimed.get("card_removals"),
        "claimed_relics": claimed.get("relics"),
        "claimed_potions": claimed.get("potions"),
        "skipped_total": skipped.get("total"),
        "skipped_gold": skipped.get("gold"),
        "skipped_primary_card_groups": skipped.get("primary_card_group"),
        "skipped_card_groups": skipped.get("card_groups"),
        "skipped_fixed_cards": skipped.get("fixed_cards"),
        "skipped_card_removals": skipped.get("card_removals"),
        "skipped_relics": skipped.get("relics"),
        "skipped_potions": skipped.get("potions"),
    }
    return tuple(_number(values.get(key)) for key in REWARD_PLAN_KEYS)


def route_plan_vector(state_payload: Mapping[str, Any]) -> tuple[float, ...]:
    """Return a stable numeric route-planning vector ordered by ROUTE_PLAN_KEYS."""

    summary = route_plan_summary(state_payload)
    styles = _mapping(summary.get("styles"))
    map_effects = _mapping(summary.get("map_effects"))
    values: dict[str, Any] = {
        "route_open": summary.get("route_open"),
        "reachable_path_count": summary.get("reachable_path_count"),
        "reachable_next_node_count": len(_sequence(summary.get("reachable_next_node_ids"))),
        "boss_path_count": summary.get("boss_path_count"),
        "boss_path_fraction": summary.get("boss_path_fraction"),
        "next_act_map_change_pending": map_effects.get("next_act_map_change_pending"),
        "golden_compass_act2_map": map_effects.get("golden_compass_act2_map"),
        "spoils_map_pending": map_effects.get("spoils_map_pending"),
        "spoils_map_target_act": map_effects.get("spoils_map_target_act"),
        "spoils_map_target_known": map_effects.get("spoils_map_target_known"),
    }
    for style in _ROUTE_STYLES:
        style_summary = _mapping(styles.get(style))
        counts = _mapping(style_summary.get("counts"))
        values[f"{style}_score"] = style_summary.get("score")
        values[f"{style}_elites"] = counts.get("elites")
        values[f"{style}_rests"] = counts.get("rests")
        values[f"{style}_shops"] = counts.get("shops")
        values[f"{style}_fights"] = counts.get("fights")
    return tuple(_number(values.get(key)) for key in ROUTE_PLAN_KEYS)


def _available_reward_choices(reward: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    metadata = _mapping(reward.get("metadata"))
    removed_card_instance_ids = set(
        _str_sequence(metadata.get("optional_removed_card_instance_ids"))
    )
    optional_remove_count = _int(metadata.get("optional_remove_card_count"))
    if optional_remove_count > len(removed_card_instance_ids):
        candidate_instance_ids = _str_sequence(metadata.get("optional_remove_card_instance_ids"))
        candidate_card_ids = _str_sequence(metadata.get("optional_remove_card_ids"))
        remaining_candidates = [
            (instance_id, candidate_card_ids[index] if index < len(candidate_card_ids) else "")
            for index, instance_id in enumerate(candidate_instance_ids)
            if instance_id not in removed_card_instance_ids
        ]
        for index, (instance_id, card_id) in enumerate(remaining_candidates):
            choices.append(
                {
                    **_choice(
                        "card_removal",
                        card_id or instance_id,
                        f"reward:remove_card:{index}",
                        "card_removal",
                        len(remaining_candidates),
                        index,
                    ),
                    "card_instance_id": instance_id,
                }
            )
    if (
        _int(reward.get("gold")) > 0
        and not reward.get("gold_claimed")
        and not reward.get("gold_skipped")
    ):
        choices.append(
            _choice(
                "gold",
                "gold",
                "reward:gold",
                "gold",
                1,
                0,
                amount=_int(reward.get("gold")),
            )
        )
    if (
        reward.get("relic_id")
        and not reward.get("relic_claimed")
        and not reward.get("relic_skipped")
    ):
        choices.append(
            _choice(
                "relic",
                str(reward.get("relic_id")),
                "reward:relic",
                "relic:single",
                1,
                0,
            )
        )

    closed_relic_ids = set(_str_sequence(reward.get("claimed_relic_ids"))) | set(
        _str_sequence(reward.get("skipped_relic_ids"))
    )
    for index, relic_id in enumerate(_sequence(reward.get("relic_ids"))):
        if str(relic_id) not in closed_relic_ids:
            choices.append(
                _choice(
                    "relic",
                    str(relic_id),
                    f"reward:relic:{index}",
                    f"relic:{index}",
                    1,
                    index,
                )
            )

    if (
        _sequence(reward.get("card_options"))
        and not reward.get("card_claimed")
        and not reward.get("card_skipped")
    ):
        group = _sequence(reward.get("card_options"))
        for index, card_id in enumerate(group):
            choices.append(
                _choice(
                    "card",
                    str(card_id),
                    f"reward:card:{index}",
                    "card_options",
                    len(group),
                    index,
                    exclusive=True,
                )
            )

    closed_fixed_cards = set(_int_sequence(reward.get("claimed_card_indices"))) | set(
        _int_sequence(reward.get("skipped_card_indices"))
    )
    for index, card_id in enumerate(_sequence(reward.get("card_ids"))):
        if index not in closed_fixed_cards:
            choices.append(
                _choice(
                    "fixed_card",
                    str(card_id),
                    f"reward:fixed_card:{index}",
                    f"fixed_card:{index}",
                    1,
                    index,
                )
            )

    closed_card_groups = set(_int_sequence(reward.get("claimed_card_option_group_indices"))) | set(
        _int_sequence(reward.get("skipped_card_option_group_indices"))
    )
    for group_index, group in enumerate(_sequence(reward.get("card_option_groups"))):
        if group_index in closed_card_groups:
            continue
        group_items = _sequence(group)
        for card_index, card_id in enumerate(group_items):
            choices.append(
                _choice(
                    "card_group",
                    str(card_id),
                    f"reward:card_group:{group_index}:{card_index}",
                    f"card_group:{group_index}",
                    len(group_items),
                    card_index,
                    group_index=group_index,
                    exclusive=True,
                )
            )

    if (
        reward.get("potion_id")
        and not reward.get("potion_claimed")
        and not reward.get("potion_skipped")
    ):
        choices.append(
            _choice(
                "potion",
                str(reward.get("potion_id")),
                "reward:potion",
                "potion:single",
                1,
                0,
            )
        )

    closed_potions = set(_int_sequence(reward.get("claimed_potion_indices"))) | set(
        _int_sequence(reward.get("skipped_potion_indices"))
    )
    for index, potion_id in enumerate(_sequence(reward.get("potion_ids"))):
        if index not in closed_potions:
            choices.append(
                _choice(
                    "potion",
                    str(potion_id),
                    f"reward:potion:{index}",
                    f"potion:{index}",
                    1,
                    index,
                )
            )
    return choices


def _choice(
    kind: str,
    content_id: str,
    target_id: str,
    selection_set_id: str,
    selection_set_size: int,
    position: int,
    *,
    amount: int = 0,
    group_index: int | None = None,
    exclusive: bool = False,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "content_id": content_id,
        "target_id": target_id,
        "selection_set_id": selection_set_id,
        "selection_set_size": selection_set_size,
        "position": position,
        "amount": amount,
        "group_index": group_index,
        "exclusive_within_set": exclusive,
    }


def _selection_sets(choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for choice in choices:
        by_id.setdefault(str(choice.get("selection_set_id", "")), []).append(choice)
    sets: list[dict[str, Any]] = []
    for selection_set_id in sorted(by_id):
        set_choices = by_id[selection_set_id]
        sets.append(
            {
                "selection_set_id": selection_set_id,
                "kind": _selection_set_kind(selection_set_id),
                "available_count": len(set_choices),
                "selection_set_size": max(
                    _int(choice.get("selection_set_size")) for choice in set_choices
                ),
                "content_ids": [str(choice.get("content_id", "")) for choice in set_choices],
                "target_ids": [str(choice.get("target_id", "")) for choice in set_choices],
                "exclusive_within_set": any(
                    bool(choice.get("exclusive_within_set")) for choice in set_choices
                ),
            }
        )
    return sets


def _selection_set_kind(selection_set_id: str) -> str:
    if selection_set_id == "card_options":
        return "card_options"
    if ":" in selection_set_id:
        return selection_set_id.split(":", 1)[0]
    return selection_set_id


def _reward_choice_counts(choices: list[dict[str, Any]]) -> dict[str, int]:
    selection_sets = {str(choice.get("selection_set_id", "")) for choice in choices}
    card_sets = {
        str(choice.get("selection_set_id", ""))
        for choice in choices
        if str(choice.get("kind", "")).startswith("card")
    }
    return {
        "total": len(choices),
        "selection_sets": len(selection_sets),
        "cards": sum(1 for choice in choices if choice.get("kind") == "card"),
        "card_groups": sum(1 for item in card_sets if item.startswith("card_group")),
        "fixed_cards": sum(1 for choice in choices if choice.get("kind") == "fixed_card"),
        "card_removals": sum(1 for choice in choices if choice.get("kind") == "card_removal"),
        "relics": sum(1 for choice in choices if choice.get("kind") == "relic"),
        "potions": sum(1 for choice in choices if choice.get("kind") == "potion"),
        "gold": sum(1 for choice in choices if choice.get("kind") == "gold"),
    }


def _reward_claimed_counts(reward: Mapping[str, Any]) -> dict[str, int]:
    metadata = _mapping(reward.get("metadata"))
    counts = {
        "gold": int(bool(reward.get("gold_claimed", False))),
        "primary_card_group": int(bool(reward.get("card_claimed", False))),
        "card_groups": len(_int_sequence(reward.get("claimed_card_option_group_indices"))),
        "fixed_cards": len(_int_sequence(reward.get("claimed_card_indices"))),
        "card_removals": len(
            _str_sequence(metadata.get("optional_removed_card_instance_ids"))
        ),
        "relics": int(bool(reward.get("relic_claimed", False)))
        + len(_str_sequence(reward.get("claimed_relic_ids"))),
        "potions": int(bool(reward.get("potion_claimed", False)))
        + len(_int_sequence(reward.get("claimed_potion_indices"))),
    }
    counts["total"] = sum(counts.values())
    return counts


def _reward_skipped_counts(reward: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "gold": int(bool(reward.get("gold_skipped", False))),
        "primary_card_group": int(bool(reward.get("card_skipped", False))),
        "card_groups": len(_int_sequence(reward.get("skipped_card_option_group_indices"))),
        "fixed_cards": len(_int_sequence(reward.get("skipped_card_indices"))),
        "card_removals": 0,
        "relics": int(bool(reward.get("relic_skipped", False)))
        + len(_str_sequence(reward.get("skipped_relic_ids"))),
        "potions": int(bool(reward.get("potion_skipped", False)))
        + len(_int_sequence(reward.get("skipped_potion_indices"))),
    }
    counts["total"] = sum(counts.values())
    return counts


def _empty_reward_counts() -> dict[str, int]:
    return {
        "total": 0,
        "selection_sets": 0,
        "cards": 0,
        "card_groups": 0,
        "fixed_cards": 0,
        "card_removals": 0,
        "relics": 0,
        "potions": 0,
        "gold": 0,
    }


def _empty_claim_skip_counts() -> dict[str, int]:
    return {
        "total": 0,
        "gold": 0,
        "primary_card_group": 0,
        "card_groups": 0,
        "fixed_cards": 0,
        "card_removals": 0,
        "relics": 0,
        "potions": 0,
    }


def _reward_group_state(
    *,
    primary_card: bool = False,
    card_groups: tuple[int, ...] = (),
    fixed_cards: tuple[int, ...] = (),
    relics: tuple[str, ...] = (),
    potions: tuple[int, ...] = (),
) -> dict[str, Any]:
    return {
        "primary_card": primary_card,
        "card_group_indices": list(card_groups),
        "fixed_card_indices": list(fixed_cards),
        "relic_ids": list(relics),
        "potion_indices": list(potions),
    }


def _next_route_start_ids(
    game_map: Mapping[str, Any],
    node_by_id: Mapping[str, Mapping[str, Any]],
    current_node_id: str,
) -> list[str]:
    outgoing = _map_outgoing_by_id(game_map)
    completed = set(_str_sequence(game_map.get("completed_node_ids")))
    if current_node_id and current_node_id in outgoing:
        candidates = outgoing[current_node_id]
    else:
        current_floor = min((_int(node.get("floor")) for node in node_by_id.values()), default=0)
        candidates = tuple(
            node_id
            for node_id, node in node_by_id.items()
            if _int(node.get("floor")) == current_floor and _node_kind(node) != "start"
        )
    return sorted(
        str(node_id)
        for node_id in candidates
        if str(node_id) in node_by_id and str(node_id) not in completed
    )


def _enumerate_paths(
    game_map: Mapping[str, Any],
    node_by_id: Mapping[str, Mapping[str, Any]],
    start_node_id: str,
    *,
    max_depth: int = 20,
    max_paths: int = 128,
) -> list[list[str]]:
    outgoing = _map_outgoing_by_id(game_map)
    completed = set(_str_sequence(game_map.get("completed_node_ids")))
    paths: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(start_node_id, [start_node_id])]
    while stack and len(paths) < max_paths:
        node_id, path = stack.pop()
        if len(path) >= max_depth or _node_kind(node_by_id.get(node_id, {})) == "boss":
            paths.append(path)
            continue
        next_ids = [
            next_id
            for next_id in outgoing.get(node_id, ())
            if next_id in node_by_id and next_id not in completed and next_id not in path
        ]
        if not next_ids:
            paths.append(path)
            continue
        for next_id in sorted(next_ids, reverse=True):
            stack.append((next_id, [*path, next_id]))
    return paths


def _path_summary(path: list[str], node_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    kinds = [_node_kind(node_by_id.get(node_id, {})) for node_id in path]
    counts = {
        "elites": kinds.count("elite"),
        "rests": kinds.count("rest"),
        "shops": kinds.count("shop"),
        "events": kinds.count("event"),
        "monsters": kinds.count("monster"),
        "treasures": kinds.count("treasure"),
        "bosses": kinds.count("boss"),
    }
    counts["fights"] = counts["monsters"] + counts["elites"] + counts["bosses"]
    return {
        "path": list(path),
        "first_node_id": path[0] if path else "",
        "depth": len(path),
        "counts": counts,
        "ends_at_boss": bool(kinds and kinds[-1] == "boss"),
    }


def _best_style_path(style: str, path_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not path_summaries:
        return _empty_style_summary(style)
    scored = [
        (_style_score(style, _mapping(path.get("counts"))), path)
        for path in path_summaries
    ]
    score, path = max(
        scored,
        key=lambda item: (
            item[0],
            int(bool(item[1].get("ends_at_boss"))),
            -_int(item[1].get("depth")),
            "|".join(str(node_id) for node_id in _sequence(item[1].get("path"))),
        ),
    )
    return {
        "style": style,
        "score": score,
        "first_node_id": str(path.get("first_node_id", "")),
        "path": list(_sequence(path.get("path"))),
        "ends_at_boss": bool(path.get("ends_at_boss", False)),
        "counts": dict(_mapping(path.get("counts"))),
    }


def _style_score(style: str, counts: Mapping[str, Any]) -> float:
    elites = _number(counts.get("elites"))
    rests = _number(counts.get("rests"))
    shops = _number(counts.get("shops"))
    monsters = _number(counts.get("monsters"))
    treasures = _number(counts.get("treasures"))
    bosses = _number(counts.get("bosses"))
    fights = _number(counts.get("fights"))
    if style == "aggressive":
        return elites * 3.0 + monsters * 1.2 + bosses * 2.0 - rests * 0.5 - shops * 0.2
    if style == "safe":
        return (
            rests * 2.0
            + shops * 0.7
            + treasures * 0.8
            - elites * 2.0
            - monsters * 0.4
            - bosses * 0.5
        )
    if style == "upgrade_heavy":
        return rests * 2.5 + treasures * 0.3 - fights * 0.2
    if style == "shop_heavy":
        return shops * 3.0 + rests * 0.2 - elites * 0.5
    if style == "elite_heavy":
        return elites * 4.0 + treasures * 0.5 + bosses * 0.5 - rests * 0.3
    return 0.0


def _empty_style_summary(style: str) -> dict[str, Any]:
    return {
        "style": style,
        "score": 0.0,
        "first_node_id": "",
        "path": [],
        "ends_at_boss": False,
        "counts": {
            "elites": 0,
            "rests": 0,
            "shops": 0,
            "events": 0,
            "monsters": 0,
            "treasures": 0,
            "bosses": 0,
            "fights": 0,
        },
    }


def _map_nodes_by_id(game_map: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = _sequence(game_map.get("nodes"))
    return {
        str(_mapping(node).get("node_id", "")): _mapping(node)
        for node in nodes
        if str(_mapping(node).get("node_id", ""))
    }


def _map_outgoing_by_id(game_map: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    outgoing: dict[str, list[str]] = {}
    for edge in _sequence(game_map.get("edges")):
        edge_map = _mapping(edge)
        from_id = str(edge_map.get("from_id", ""))
        to_id = str(edge_map.get("to_id", ""))
        if from_id and to_id:
            outgoing.setdefault(from_id, []).append(to_id)
    return {node_id: tuple(sorted(next_ids)) for node_id, next_ids in outgoing.items()}


def _node_kind(node: Mapping[str, Any]) -> str:
    kind = str(node.get("kind", "")).lower()
    return kind.rsplit(".", 1)[-1]


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or value is None or isinstance(value, Mapping):
        return ()
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def _str_sequence(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value))


def _int_sequence(value: Any) -> tuple[int, ...]:
    return tuple(_int(item) for item in _sequence(value))


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(value: Any) -> int:
    return int(_number(value))
